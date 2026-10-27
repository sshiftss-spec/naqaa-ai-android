package com.naqaa.ai

import android.graphics.Bitmap
import android.graphics.Color
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.FloatBuffer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

/**
 * Experimental full-frame NAFNet deblurring.
 * The model is applied to overlapping 256px tiles so large photos do not require one huge tensor.
 */
internal object NafNetDeblurRestorer {
    private const val MODEL_URL =
        "https://huggingface.co/opencv/deblurring_nafnet/resolve/main/deblurring_nafnet_2025may.onnx?download=true"
    private const val MIN_MODEL_BYTES = 80_000_000L
    private const val TILE = 256
    private const val STEP = 224

    fun ensureModel(filesDir: File): File {
        val destination = File(filesDir, "nafnet_deblur.onnx")
        if (destination.isFile && destination.length() >= MIN_MODEL_BYTES) return destination

        val partial = File(filesDir, "nafnet_deblur.onnx.download")
        val connection = (URL(MODEL_URL).openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 120_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "NaqaaAI-Android")
        }
        try {
            val code = connection.responseCode
            if (code != HttpURLConnection.HTTP_OK) {
                throw IllegalStateException("تعذر تنزيل النموذج (HTTP $code)")
            }
            connection.inputStream.use { input ->
                FileOutputStream(partial).use { output -> input.copyTo(output, 1024 * 1024) }
            }
            if (partial.length() < MIN_MODEL_BYTES) {
                throw IllegalStateException("ملف النموذج غير مكتمل؛ أعد المحاولة باتصال ثابت")
            }
            if (destination.exists()) destination.delete()
            if (!partial.renameTo(destination)) {
                throw IllegalStateException("تعذر حفظ ملف النموذج على الجهاز")
            }
            return destination
        } finally {
            connection.disconnect()
            if (partial.exists()) partial.delete()
        }
    }

    fun restore(source: Bitmap, modelFile: File): Bitmap {
        require(source.width > 0 && source.height > 0) { "الصورة غير صالحة" }
        val width = source.width
        val height = source.height
        val outputPixels = IntArray(width * height)
        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }

        env.createSession(modelFile.absolutePath, options).use { session ->
            val inputName = session.inputNames.firstOrNull()
                ?: throw IllegalStateException("النموذج لا يحتوي على مدخل")
            val xStarts = tileStarts(width)
            val yStarts = tileStarts(height)

            for (top in yStarts) {
                for (left in xStarts) {
                    val input = FloatArray(3 * TILE * TILE)
                    for (y in 0 until TILE) {
                        val sy = (top + y).coerceIn(0, height - 1)
                        for (x in 0 until TILE) {
                            val sx = (left + x).coerceIn(0, width - 1)
                            val color = source.getPixel(sx, sy)
                            val i = y * TILE + x
                            input[i] = Color.red(color) / 255f
                            input[TILE * TILE + i] = Color.green(color) / 255f
                            input[2 * TILE * TILE + i] = Color.blue(color) / 255f
                        }
                    }

                    OnnxTensor.createTensor(
                        env,
                        FloatBuffer.wrap(input),
                        longArrayOf(1, 3, TILE.toLong(), TILE.toLong())
                    ).use { tensor ->
                        session.run(mapOf(inputName to tensor)).use { result ->
                            @Suppress("UNCHECKED_CAST")
                            val values = result[0].value as? Array<Array<Array<FloatArray>>>
                                ?: throw IllegalStateException("صيغة خرج النموذج غير متوقعة")
                            if (values[0].size < 3 || values[0][0].size < TILE || values[0][0][0].size < TILE) {
                                throw IllegalStateException("أبعاد خرج النموذج غير متوافقة")
                            }

                            val localLeft = if (left == 0) 0 else 16
                            val localTop = if (top == 0) 0 else 16
                            val localRight = if (left + TILE >= width) TILE else TILE - 16
                            val localBottom = if (top + TILE >= height) TILE else TILE - 16
                            for (y in localTop until localBottom) {
                                val gy = top + y
                                if (gy !in 0 until height) continue
                                for (x in localLeft until localRight) {
                                    val gx = left + x
                                    if (gx !in 0 until width) continue
                                    val i = y * TILE + x
                                    val r = (values[0][0][y][x] * 255f).toInt().coerceIn(0, 255)
                                    val g = (values[0][1][y][x] * 255f).toInt().coerceIn(0, 255)
                                    val b = (values[0][2][y][x] * 255f).toInt().coerceIn(0, 255)
                                    outputPixels[gy * width + gx] = Color.rgb(r, g, b)
                                }
                            }
                        }
                    }
                }
            }
        }
        return Bitmap.createBitmap(outputPixels, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun tileStarts(size: Int): List<Int> {
        if (size <= TILE) return listOf(0)
        val starts = mutableListOf<Int>()
        var position = 0
        while (position + TILE < size) {
            starts.add(position)
            position += STEP
        }
        val last = size - TILE
        if (starts.isEmpty() || starts.last() != last) starts.add(last)
        return starts.distinct()
    }
}
