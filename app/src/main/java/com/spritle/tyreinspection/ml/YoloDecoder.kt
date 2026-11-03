package com.spritle.tyreinspection.ml

/**
 * Decodes the raw output of an Ultralytics YOLOv8 / YOLO11 TFLite export.
 *
 * The output tensor is `[1, 4 + numClasses, numAnchors]` (channel-major, the default export)
 * or `[1, numAnchors, 4 + numClasses]` (anchor-major, some converters). Each anchor holds
 * `cx, cy, w, h` followed by one score per class. Coordinates are normalized 0..1 in recent
 * exports; pixel coordinates (older exports) are rescaled using [inputSize].
 */
object YoloDecoder {

    fun decode(
        output: FloatArray,
        dim1: Int,
        dim2: Int,
        labels: List<String>,
        inputSize: Int,
        confidenceThreshold: Float,
        iouThreshold: Float,
    ): List<Detection> {
        // The class/box axis is always the smaller one (e.g. 10 vs 8400).
        val channelMajor = dim1 <= dim2
        val numChannels = if (channelMajor) dim1 else dim2
        val numAnchors = if (channelMajor) dim2 else dim1
        val numClasses = numChannels - 4
        if (numClasses <= 0) return emptyList()

        fun value(anchor: Int, channel: Int) =
            if (channelMajor) output[channel * numAnchors + anchor] else output[anchor * numChannels + channel]

        var pixelCoords = false
        val candidates = mutableListOf<Detection>()
        for (a in 0 until numAnchors) {
            var bestScore = 0f
            var bestClass = -1
            for (c in 0 until numClasses) {
                val s = value(a, 4 + c)
                if (s > bestScore) {
                    bestScore = s
                    bestClass = c
                }
            }
            if (bestClass < 0 || bestScore < confidenceThreshold) continue

            val cx = value(a, 0)
            val cy = value(a, 1)
            val w = value(a, 2)
            val h = value(a, 3)
            if (cx > 1.5f || cy > 1.5f || w > 1.5f || h > 1.5f) pixelCoords = true
            candidates += Detection(
                label = labels.getOrElse(bestClass) { "class_$bestClass" },
                confidence = bestScore,
                box = Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
            )
        }

        val scale = if (pixelCoords) 1f / inputSize else 1f
        val normalized = candidates.map { d ->
            d.copy(
                box = Box(
                    (d.box.x1 * scale).coerceIn(0f, 1f),
                    (d.box.y1 * scale).coerceIn(0f, 1f),
                    (d.box.x2 * scale).coerceIn(0f, 1f),
                    (d.box.y2 * scale).coerceIn(0f, 1f),
                )
            )
        }.filter { it.box.area > 0f }

        return nonMaxSuppression(normalized, iouThreshold)
    }

    /** Class-aware greedy NMS: a box only suppresses lower-scoring boxes of the same label. */
    fun nonMaxSuppression(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val remaining = detections.sortedByDescending { it.confidence }.toMutableList()
        val kept = mutableListOf<Detection>()
        while (remaining.isNotEmpty()) {
            val best = remaining.removeAt(0)
            kept += best
            remaining.removeAll { it.label == best.label && iou(best.box, it.box) >= iouThreshold }
        }
        return kept
    }

    fun iou(a: Box, b: Box): Float {
        val ix = maxOf(0f, minOf(a.x2, b.x2) - maxOf(a.x1, b.x1))
        val iy = maxOf(0f, minOf(a.y2, b.y2) - maxOf(a.y1, b.y1))
        val inter = ix * iy
        val union = a.area + b.area - inter
        return if (union <= 0f) 0f else inter / union
    }
}
