package com.spritle.tyreinspection.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

/**
 * Thin wrapper around a LiteRT [Interpreter] for single-image-in models.
 *
 * Handles NHWC and NCHW inputs and FLOAT32 / UINT8 / INT8 tensors (quantized models are
 * (de)quantized with the tensor's own scale and zero point), so float and int8 exports of
 * the same model work without code changes. Not thread-safe: use one instance per thread.
 */
class TfliteModel private constructor(
    private val interpreter: Interpreter,
    val labels: List<String>,
) : AutoCloseable {

    private val inputTensor = interpreter.getInputTensor(0)
    private val nchw = inputTensor.shape()[1] == 3
    val inputWidth = if (nchw) inputTensor.shape()[3] else inputTensor.shape()[2]
    val inputHeight = if (nchw) inputTensor.shape()[2] else inputTensor.shape()[1]
    val outputShape: IntArray = interpreter.getOutputTensor(0).shape()

    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(inputTensor.numBytes()).order(ByteOrder.nativeOrder())
    private val outputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(interpreter.getOutputTensor(0).numBytes()).order(ByteOrder.nativeOrder())
    private val pixels = IntArray(inputWidth * inputHeight)

    /**
     * Resizes [bitmap] to the input size, feeds it and returns output tensor 0 as floats.
     * @param pixelDivisor 1 to feed raw 0..255 values, 255 to feed 0..1 values.
     */
    fun run(bitmap: Bitmap, pixelDivisor: Float): FloatArray {
        fillInput(bitmap, pixelDivisor)
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)
        return readOutput()
    }

    private fun fillInput(bitmap: Bitmap, pixelDivisor: Float) {
        val resized = if (bitmap.width == inputWidth && bitmap.height == inputHeight) bitmap
        else Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        resized.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        if (resized !== bitmap) resized.recycle()

        val q = inputTensor.quantizationParams()
        val type = inputTensor.dataType()
        inputBuffer.rewind()

        fun put(channelValue: Int) {
            val v = channelValue / pixelDivisor
            when (type) {
                DataType.FLOAT32 -> inputBuffer.putFloat(v)
                DataType.UINT8 -> inputBuffer.put(quantize(v, q.scale, q.zeroPoint, 0, 255).toByte())
                DataType.INT8 -> inputBuffer.put(quantize(v, q.scale, q.zeroPoint, -128, 127).toByte())
                else -> error("Unsupported input type $type")
            }
        }

        if (nchw) {
            for (shift in intArrayOf(16, 8, 0)) for (p in pixels) put((p shr shift) and 0xFF)
        } else {
            for (p in pixels) {
                put((p shr 16) and 0xFF)
                put((p shr 8) and 0xFF)
                put(p and 0xFF)
            }
        }
        inputBuffer.rewind()
    }

    private fun quantize(v: Float, scale: Float, zeroPoint: Int, min: Int, max: Int): Int =
        if (scale == 0f) v.roundToInt().coerceIn(min, max)
        else (v / scale + zeroPoint).roundToInt().coerceIn(min, max)

    private fun readOutput(): FloatArray {
        val tensor = interpreter.getOutputTensor(0)
        val q = tensor.quantizationParams()
        outputBuffer.rewind()
        return when (val type = tensor.dataType()) {
            DataType.FLOAT32 -> FloatArray(tensor.numElements()).also { outputBuffer.asFloatBuffer().get(it) }
            DataType.UINT8 -> FloatArray(tensor.numElements()) { ((outputBuffer.get().toInt() and 0xFF) - q.zeroPoint) * q.scale }
            DataType.INT8 -> FloatArray(tensor.numElements()) { (outputBuffer.get().toInt() - q.zeroPoint) * q.scale }
            else -> error("Unsupported output type $type")
        }
    }

    override fun close() = interpreter.close()

    companion object {
        private const val TAG = "TfliteModel"

        /** Returns null (and logs) when the model asset is missing or invalid, so the app still runs. */
        fun loadOrNull(context: Context, modelAsset: String, labelsAsset: String): TfliteModel? = try {
            val options = Interpreter.Options().apply { numThreads = 4 }
            val interpreter = Interpreter(FileUtil.loadMappedFile(context, modelAsset), options)
            val labels = context.assets.open(labelsAsset).bufferedReader().readLines()
                .map { it.trim() }.filter { it.isNotEmpty() }
            TfliteModel(interpreter, labels)
        } catch (e: Exception) {
            Log.w(TAG, "Model $modelAsset not loaded: ${e.message}")
            null
        }
    }
}
