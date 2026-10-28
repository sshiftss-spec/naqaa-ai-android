package com.naqaa.ai

import android.graphics.Bitmap
import android.graphics.Color
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.min

/**
 * Experimental local Real-ESRGAN x4plus runner.
 * The dynamic-H/W ONNX graph accepts RGB float32 in [0,1] and emits 4x RGB.
 * Tiles are capped at 128x128 with a 16px overlap to limit peak memory on phones.
 */
internal object RealEsrganRestorer {
    private const val TILE = 128
    private const val OVERLAP = 16
    private const val STEP = TILE - 2 * OVERLAP
    private const val SCALE = 4

    fun restore(source: Bitmap, model: File): Bitmap {
        require(model.exists() && model.length() > 60_000_000L) { "ملف Real-ESRGAN غير مكتمل؛ أعد تنزيل النموذج." }
        // Bound working memory for large camera photos. Output is 4x this working image.
        val maxSide = 1024
        val ratio = min(1f, maxSide.toFloat() / maxOf(source.width, source.height))
        val input = if (ratio < 1f) Bitmap.createScaledBitmap(
            source, (source.width * ratio).toInt().coerceAtLeast(1),
            (source.height * ratio).toInt().coerceAtLeast(1), true
        ) else source
        val outW = input.width * SCALE
        val outH = input.height * SCALE
        val output = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
        env.createSession(model.absolutePath, options).use { session ->
            val inputName = session.inputNames.first()
            val inputType = session.inputInfo[inputName]!!.info as ai.onnxruntime.TensorInfo
            require(inputType.shape.size == 4 && inputType.shape[1] == 3L) {
                "بنية نموذج Real-ESRGAN غير متوافقة (المطلوب NCHW RGB)."
            }
            var top = 0
            while (top < input.height) {
                var left = 0
                while (left < input.width) {
                    val tileW = min(TILE, input.width - left)
                    val tileH = min(TILE, input.height - top)
                    val tile = Bitmap.createBitmap(TILE, TILE, Bitmap.Config.ARGB_8888)
                    val canvas = android.graphics.Canvas(tile)
                    canvas.drawBitmap(input, -left.toFloat(), -top.toFloat(), android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG))
                    if (tileW < TILE || tileH < TILE) {
                        // Canvas draws the available edge; fill any uncovered padding with edge pixels.
                        val pixels = IntArray(TILE * TILE)
                        tile.getPixels(pixels, 0, TILE, 0, 0, TILE, TILE)
                        for (y in 0 until TILE) for (x in 0 until TILE) {
                            val sx = x.coerceAtMost(tileW - 1)
                            val sy = y.coerceAtMost(tileH - 1)
                            pixels[y * TILE + x] = pixels[sy * TILE + sx]
                        }
                        tile.setPixels(pixels, 0, TILE, 0, 0, TILE, TILE)
                    }
                    val rgb = FloatArray(3 * TILE * TILE)
                    val pixels = IntArray(TILE * TILE)
                    tile.getPixels(pixels, 0, TILE, 0, 0, TILE, TILE)
                    for (i in pixels.indices) {
                        rgb[i] = Color.red(pixels[i]) / 255f
                        rgb[TILE * TILE + i] = Color.green(pixels[i]) / 255f
                        rgb[2 * TILE * TILE + i] = Color.blue(pixels[i]) / 255f
                    }
                    val restored = OnnxTensor.createTensor(
                        env, FloatBuffer.wrap(rgb), longArrayOf(1, 3, TILE.toLong(), TILE.toLong())
                    ).use { tensor ->
                        session.run(mapOf(inputName to tensor)).use { result ->
                            @Suppress("UNCHECKED_CAST")
                            val planes = result[0].value as Array<Array<Array<FloatArray>>>
                            val data = IntArray(TILE * SCALE * TILE * SCALE)
                            val outSize = TILE * SCALE
                            for (y in 0 until outSize) for (x in 0 until outSize) {
                                val r = (planes[0][0][y][x] * 255f).toInt().coerceIn(0, 255)
                                val g = (planes[0][1][y][x] * 255f).toInt().coerceIn(0, 255)
                                val b = (planes[0][2][y][x] * 255f).toInt().coerceIn(0, 255)
                                data[y * outSize + x] = Color.rgb(r, g, b)
                            }
                            Bitmap.createBitmap(data, outSize, outSize, Bitmap.Config.ARGB_8888)
                        }
                    }
                    val copyLeft = if (left == 0) 0 else OVERLAP * SCALE
                    val copyTop = if (top == 0) 0 else OVERLAP * SCALE
                    val copyRight = min(tileW * SCALE, (TILE - if (left + tileW < input.width) OVERLAP else 0) * SCALE)
                    val copyBottom = min(tileH * SCALE, (TILE - if (top + tileH < input.height) OVERLAP else 0) * SCALE)
                    val destX = left * SCALE + copyLeft
                    val destY = top * SCALE + copyTop
                    val copyW = (copyRight - copyLeft).coerceAtLeast(0)
                    val copyH = (copyBottom - copyTop).coerceAtLeast(0)
                    if (copyW > 0 && copyH > 0) {
                        val region = Bitmap.createBitmap(restored, copyLeft, copyTop, copyW, copyH)
                        android.graphics.Canvas(output).drawBitmap(region, destX.toFloat(), destY.toFloat(), null)
                        region.recycle()
                    }
                    tile.recycle()
                    restored.recycle()
                    left += STEP
                }
                top += STEP
            }
        }
        if (input !== source) input.recycle()
        return output
    }
}
