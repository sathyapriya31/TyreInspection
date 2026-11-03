package com.spritle.tyreinspection.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.spritle.tyreinspection.R
import com.spritle.tyreinspection.data.InspectionRepository
import com.spritle.tyreinspection.databinding.ActivityMainBinding
import com.spritle.tyreinspection.ml.ImageUtils
import com.spritle.tyreinspection.ml.InspectionResult
import com.spritle.tyreinspection.ml.TyreInspector
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Live camera screen. Runs the models on every preview frame (latest-only backpressure)
 * to show defects in real time, and captures a full-resolution photo for the detailed report.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var repository: InspectionRepository

    /** All model work happens on this single thread; [inspector] is only touched from it. */
    private lateinit var cameraExecutor: ExecutorService
    private var inspector: TyreInspector? = null

    private var camera: Camera? = null
    private var imageCapture: ImageCapture? = null
    @Volatile private var busy = false

    private val requestCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera()
        else Toast.makeText(this, R.string.camera_permission_needed, Toast.LENGTH_LONG).show()
    }

    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let(::inspectFromGallery)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        repository = InspectionRepository(this)
        cameraExecutor = Executors.newSingleThreadExecutor()
        cameraExecutor.execute {
            val loaded = TyreInspector(this)
            inspector = loaded
            runOnUiThread { showModelStatus(loaded) }
        }

        binding.captureButton.setOnClickListener { capture() }
        binding.galleryButton.setOnClickListener {
            pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        binding.historyButton.setOnClickListener { startActivity(HistoryActivity.intent(this)) }
        binding.torchSwitch.setOnCheckedChangeListener { _, on -> camera?.cameraControl?.enableTorch(on) }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestCamera.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onResume() {
        super.onResume()
        setBusy(false)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.execute { inspector?.close() }
        cameraExecutor.shutdown()
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            // Same 4:3 aspect for preview and analysis so overlay boxes align with the preview.
            val resolution = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .build()

            val preview = Preview.Builder().setResolutionSelector(resolution).build()
                .also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }
            val capture = ImageCapture.Builder()
                .setResolutionSelector(resolution)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolution)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(cameraExecutor, ::analyzeFrame) }

            provider.unbindAll()
            camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture, analysis)
            imageCapture = capture
            binding.torchSwitch.isEnabled = camera?.cameraInfo?.hasFlashUnit() == true
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(image: ImageProxy) {
        val rotation = image.imageInfo.rotationDegrees
        val frame = image.use { it.toBitmap() }
        val model = inspector
        if (busy || model == null) return

        val upright = ImageUtils.rotate(frame, rotation)
        val result = model.inspect(upright)
        runOnUiThread {
            if (busy) return@runOnUiThread
            binding.overlay.setResults(result.defects, upright.width, upright.height)
            showLiveResult(result)
        }
    }

    private fun showLiveResult(result: InspectionResult) {
        binding.liveVerdict.setText(result.verdict.labelRes)
        binding.liveVerdict.setTextColor(ContextCompat.getColor(this, result.verdict.colorRes))
        val condition = result.condition?.let {
            getString(R.string.condition_value, it.label, (it.confidence * 100).toInt())
        } ?: getString(R.string.condition_unavailable)
        binding.liveDetail.text = getString(R.string.live_status, condition, result.defects.size)
        binding.liveDetail.visibility = View.VISIBLE
    }

    private fun showModelStatus(inspector: TyreInspector) {
        val message = when {
            !inspector.classifierAvailable && !inspector.detectorAvailable -> getString(R.string.models_missing)
            !inspector.detectorAvailable -> getString(R.string.models_partial, getString(R.string.model_classifier))
            !inspector.classifierAvailable -> getString(R.string.models_partial, getString(R.string.model_detector))
            else -> null
        }
        binding.modelBanner.text = message
        binding.modelBanner.visibility = if (message == null) View.GONE else View.VISIBLE
    }

    private fun capture() {
        val capture = imageCapture ?: return
        setBusy(true)
        capture.takePicture(cameraExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val rotation = image.imageInfo.rotationDegrees
                val bitmap = image.use { it.toBitmap() }
                openResult(ImageUtils.scaleDown(ImageUtils.rotate(bitmap, rotation), MAX_PHOTO_SIDE))
            }

            override fun onError(e: ImageCaptureException) {
                runOnUiThread {
                    Toast.makeText(this@MainActivity, getString(R.string.capture_failed, e.message), Toast.LENGTH_LONG).show()
                    setBusy(false)
                }
            }
        })
    }

    private fun inspectFromGallery(uri: Uri) {
        setBusy(true)
        cameraExecutor.execute {
            val bitmap = runCatching { ImageUtils.decodeUri(this, uri, MAX_PHOTO_SIDE) }.getOrNull()
            if (bitmap == null) {
                runOnUiThread {
                    Toast.makeText(this, R.string.import_failed, Toast.LENGTH_LONG).show()
                    setBusy(false)
                }
            } else {
                openResult(bitmap)
            }
        }
    }

    /** Called on [cameraExecutor]: persists the photo, then hands off to the report screen. */
    private fun openResult(bitmap: Bitmap) {
        val id = repository.newId()
        repository.saveImage(id, bitmap)
        runOnUiThread { startActivity(ResultActivity.newInspection(this, id, selectedPosition())) }
    }

    private fun selectedPosition() = when (binding.positionGroup.checkedButtonId) {
        R.id.posFR -> "FR"
        R.id.posRL -> "RL"
        R.id.posRR -> "RR"
        R.id.posSpare -> "SPARE"
        else -> "FL"
    }

    private fun setBusy(value: Boolean) {
        busy = value
        binding.progress.visibility = if (value) View.VISIBLE else View.GONE
        binding.captureButton.isEnabled = !value
        binding.galleryButton.isEnabled = !value
        if (value) binding.overlay.clear()
    }

    companion object {
        /** Longest side of stored photos; enough detail for the models and a readable report. */
        private const val MAX_PHOTO_SIDE = 1600
    }
}
