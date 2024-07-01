package com.example.livenessdetection

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.livenessdetection.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private lateinit var selectImageButton: Button
    private lateinit var makePredictionButton: Button
    private lateinit var imageView: ImageView
    private lateinit var textView: TextView
    private lateinit var cameraButton: Button
    private lateinit var bitmap: Bitmap
    private lateinit var model: Model

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        selectImageButton = findViewById(R.id.button)
        makePredictionButton = findViewById(R.id.button2)
        imageView = findViewById(R.id.imageView2)
        textView = findViewById(R.id.textView)
        cameraButton = findViewById(R.id.camerabtn)

        // Handle permissions
        checkAndRequestPermissions()

        // Initialize the TensorFlow Lite model
        model = Model.newInstance(this)

        selectImageButton.setOnClickListener {
            Log.d("MainActivity", "Select Image button pressed")
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"
            startActivityForResult(intent, REQUEST_CODE_SELECT_IMAGE)
        }

        makePredictionButton.setOnClickListener {
            if (!::bitmap.isInitialized) {
                Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 128, 128, true)

            // Convert Bitmap to ByteBuffer
            val byteBuffer = convertBitmapToByteBuffer(resizedBitmap)

            // Create input TensorBuffer with FLOAT32 data type
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 128, 128, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(byteBuffer)

            // Run inference
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            // Process the output
            val result = if (outputFeature0.floatArray[0] < 0.5) "Live" else "Spoof"

            // Ensure UI updates are run on the main thread
            runOnUiThread {
                textView.text = "Prediction: $result"
            }
        }

        cameraButton.setOnClickListener {
            val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(cameraIntent, REQUEST_CODE_CAMERA)
        }
    }

    private fun checkAndRequestPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQUEST_CODE_PERMISSIONS)
        } else {
            Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show()
        }
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(1 * 128 * 128 * 3 * 4) // 1 image, 128x128 pixels, 3 channels (RGB), FLOAT32
        byteBuffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(128 * 128)
        bitmap.getPixels(pixels, 0, 128, 0, 0, 128, 128)
        var pixel = 0
        for (i in 0 until 128) {
            for (j in 0 until 128) {
                val pixelValue = pixels[pixel++]
                byteBuffer.putFloat((pixelValue shr 16 and 0xFF) / 255.0f) // Red channel
                byteBuffer.putFloat((pixelValue shr 8 and 0xFF) / 255.0f)  // Green channel
                byteBuffer.putFloat((pixelValue and 0xFF) / 255.0f)         // Blue channel
            }
        }
        return byteBuffer
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == REQUEST_CODE_SELECT_IMAGE && resultCode == Activity.RESULT_OK && data != null) {
            val selectedImageUri: Uri? = data.data
            imageView.setImageURI(selectedImageUri)
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, selectedImageUri)
        }

        if (requestCode == REQUEST_CODE_CAMERA && resultCode == Activity.RESULT_OK && data != null) {
            bitmap = data.extras?.get("data") as Bitmap
            imageView.setImageBitmap(bitmap)
        }
    }

    override fun onDestroy() {
        model.close()
        super.onDestroy()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 100
        private const val REQUEST_CODE_SELECT_IMAGE = 250
        private const val REQUEST_CODE_CAMERA = 200
    }
}
