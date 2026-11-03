package com.spritle.tyreinspection.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class YoloDecoderTest {

    private val labels = listOf("bulge", "crack")

    /**
     * Builds a channel-major [4 + classes, ANCHORS] tensor from per-anchor rows, padding with
     * empty anchors: the decoder tells layouts apart by anchors outnumbering channels.
     */
    private fun channelMajor(rows: List<FloatArray>): FloatArray {
        val channels = rows.first().size
        val padded = rows + List(ANCHORS - rows.size) { FloatArray(channels) }
        return FloatArray(channels * ANCHORS) { i -> padded[i % ANCHORS][i / ANCHORS] }
    }

    private companion object {
        const val ANCHORS = 8
    }

    private fun decode(output: FloatArray, dim1: Int, dim2: Int) =
        YoloDecoder.decode(output, dim1, dim2, labels, inputSize = 640, confidenceThreshold = 0.4f, iouThreshold = 0.5f)

    @Test
    fun channelMajor_decodesBestClassAndBox() {
        val rows = listOf(
            floatArrayOf(0.5f, 0.5f, 0.2f, 0.4f, 0.1f, 0.9f), // crack
            floatArrayOf(0.2f, 0.2f, 0.1f, 0.1f, 0.2f, 0.1f), // below threshold
        )
        val result = decode(channelMajor(rows), 6, ANCHORS)
        assertEquals(1, result.size)
        assertEquals("crack", result[0].label)
        assertEquals(0.4f, result[0].box.x1, 1e-5f)
        assertEquals(0.7f, result[0].box.y2, 1e-5f)
    }

    @Test
    fun anchorMajor_andPixelCoordinates_areNormalized() {
        // Real exports have far more anchors than channels; pad to 8 anchors x 6 channels.
        val output = FloatArray(8 * 6).also {
            floatArrayOf(320f, 320f, 64f, 64f, 0.8f, 0.1f).copyInto(it)
        }
        val result = decode(output, 8, 6)
        assertEquals(1, result.size)
        assertEquals("bulge", result[0].label)
        assertEquals(0.45f, result[0].box.x1, 1e-5f)
        assertEquals(0.55f, result[0].box.x2, 1e-5f)
    }

    @Test
    fun nms_removesOverlapsOfSameClassOnly() {
        val rows = listOf(
            floatArrayOf(0.5f, 0.5f, 0.2f, 0.2f, 0.9f, 0f),
            floatArrayOf(0.51f, 0.5f, 0.2f, 0.2f, 0.7f, 0f), // overlapping bulge -> suppressed
            floatArrayOf(0.5f, 0.5f, 0.2f, 0.2f, 0f, 0.6f),  // same place but crack -> kept
        )
        val result = decode(channelMajor(rows), 6, ANCHORS)
        assertEquals(listOf("bulge", "crack"), result.map { it.label })
    }
}
