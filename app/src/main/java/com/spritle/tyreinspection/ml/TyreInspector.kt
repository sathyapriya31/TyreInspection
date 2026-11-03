package com.spritle.tyreinspection.ml

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import kotlin.math.exp

/**
 * Runs the two on-device models on a tyre photo and combines them into an [InspectionResult].
 *
 * - Condition classifier (`tyre_condition.tflite`): whole-image class such as good / defective.
 *   Expects raw 0..255 RGB input (Keras MobileNetV3 with built-in preprocessing).
 * - Defect detector (`tyre_defects.tflite`): YOLOv8/YOLO11 export that localises damage.
 *   Expects 0..1 RGB input.
 *
 * Either model may be absent; the verdict then uses whatever is available.
 * Not thread-safe: create one instance per worker thread and [close] it when done.
 */
class TyreInspector(context: Context) : AutoCloseable {

    private val classifier = TfliteModel.loadOrNull(context, CLASSIFIER_MODEL, CLASSIFIER_LABELS)
    private val detector = TfliteModel.loadOrNull(context, DETECTOR_MODEL, DETECTOR_LABELS)

    val classifierAvailable get() = classifier != null
    val detectorAvailable get() = detector != null

    fun inspect(bitmap: Bitmap, detectionThreshold: Float = DETECTION_THRESHOLD): InspectionResult {
        val start = SystemClock.uptimeMillis()
        val quality = ImageUtils.measureQuality(bitmap)
        val condition = classify(bitmap).firstOrNull()
        val defects = detect(bitmap, detectionThreshold)
        val (verdict, advice) = VerdictRules.evaluate(
            condition, defects, quality, classifierAvailable, detectorAvailable
        )
        return InspectionResult(
            verdict = verdict,
            condition = condition,
            defects = defects,
            quality = quality,
            recommendations = advice,
            inferenceMs = SystemClock.uptimeMillis() - start,
            classifierAvailable = classifierAvailable,
            detectorAvailable = detectorAvailable,
        )
    }

    /** All classes sorted by confidence, highest first; empty when the classifier is absent. */
    fun classify(bitmap: Bitmap): List<ClassScore> {
        val model = classifier ?: return emptyList()
        val raw = model.run(bitmap, pixelDivisor = 1f)
        val probs = if (looksLikeProbabilities(raw)) raw else softmax(raw)
        return probs.mapIndexed { i, p -> ClassScore(model.labels.getOrElse(i) { "class_$i" }, p) }
            .sortedByDescending { it.confidence }
    }

    fun detect(bitmap: Bitmap, threshold: Float = DETECTION_THRESHOLD): List<Detection> {
        val model = detector ?: return emptyList()
        val output = model.run(bitmap, pixelDivisor = 255f)
        val shape = model.outputShape
        return YoloDecoder.decode(
            output = output,
            dim1 = shape[1],
            dim2 = shape[2],
            labels = model.labels,
            inputSize = maxOf(model.inputWidth, model.inputHeight),
            confidenceThreshold = threshold,
            iouThreshold = IOU_THRESHOLD,
        )
    }

    override fun close() {
        classifier?.close()
        detector?.close()
    }

    private fun looksLikeProbabilities(v: FloatArray) =
        v.all { it in 0f..1f } && kotlin.math.abs(v.sum() - 1f) < 0.05f

    private fun softmax(v: FloatArray): FloatArray {
        val max = v.max()
        val e = v.map { exp((it - max).toDouble()) }
        val sum = e.sum()
        return FloatArray(v.size) { (e[it] / sum).toFloat() }
    }

    companion object {
        const val CLASSIFIER_MODEL = "tyre_condition.tflite"
        const val CLASSIFIER_LABELS = "condition_labels.txt"
        const val DETECTOR_MODEL = "tyre_defects.tflite"
        const val DETECTOR_LABELS = "defect_labels.txt"

        const val DETECTION_THRESHOLD = 0.4f
        const val IOU_THRESHOLD = 0.5f
    }
}
