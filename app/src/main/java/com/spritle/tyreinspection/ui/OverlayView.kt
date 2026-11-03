package com.spritle.tyreinspection.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.spritle.tyreinspection.R
import com.spritle.tyreinspection.ml.Detection
import com.spritle.tyreinspection.ml.Severity

/**
 * Draws defect boxes over an image shown with fit-center scaling (PreviewView `fitCenter`
 * or ImageView `fitCenter`), so normalized boxes line up with the picture underneath.
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var detections: List<Detection> = emptyList()
    private var imageWidth = 0
    private var imageHeight = 0

    private val density = resources.displayMetrics.density
    private val boxPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 3 * density }
    private val labelBgPaint = Paint().apply { style = Paint.Style.FILL }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.white)
        textSize = 13 * density
    }
    private val rect = RectF()

    fun setResults(detections: List<Detection>, imageWidth: Int, imageHeight: Int) {
        this.detections = detections
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight
        invalidate()
    }

    fun clear() = setResults(emptyList(), 0, 0)

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (imageWidth == 0 || imageHeight == 0) return

        val scale = minOf(width / imageWidth.toFloat(), height / imageHeight.toFloat())
        val drawnW = imageWidth * scale
        val drawnH = imageHeight * scale
        val left = (width - drawnW) / 2
        val top = (height - drawnH) / 2

        for (d in detections) {
            val color = ContextCompat.getColor(context, severityColor(d.severity))
            boxPaint.color = color
            labelBgPaint.color = color
            rect.set(
                left + d.box.x1 * drawnW, top + d.box.y1 * drawnH,
                left + d.box.x2 * drawnW, top + d.box.y2 * drawnH,
            )
            canvas.drawRect(rect, boxPaint)

            val text = "${d.label.replace('_', ' ')} ${(d.confidence * 100).toInt()}%"
            val textW = textPaint.measureText(text)
            val textH = textPaint.textSize
            val labelTop = (rect.top - textH - 8).coerceAtLeast(top)
            canvas.drawRect(rect.left, labelTop, rect.left + textW + 12, labelTop + textH + 8, labelBgPaint)
            canvas.drawText(text, rect.left + 6, labelTop + textH, textPaint)
        }
    }

    private fun severityColor(s: Severity) = when (s) {
        Severity.CRITICAL -> R.color.verdict_fail
        Severity.MAJOR -> R.color.verdict_attention
        Severity.MINOR -> R.color.brand_accent
    }
}
