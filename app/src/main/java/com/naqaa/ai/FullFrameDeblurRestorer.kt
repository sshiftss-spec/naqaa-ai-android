package com.naqaa.ai

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

/**
 * Experimental whole-frame motion deblurring using RT-Focuser ONNX.
 * The released model is trained for GoPro-style motion blur and expects 256x256 RGB [0,1].
 * Tiled inference preserves the source resolution; it does not downscale the entire image.
 */
internal object FullFrameDeblurRestorer {
    private const val TILE = 256
    private const val MODEL_URL =
        "https://raw.githubusercontent.com/ReaganWu/RT-Focuser/main/Pretrained_Weights/rt_focuser_wint8_afp32.onnx"
    private const val MIN_MODEL_BYTES = 20_000_000L

    fun ensureModel(model: File) {
        if (model.exists() && model.length() >= MIN_MODEL_BYTES) return
        model.parentFile?.mkdirs()
        val partial = File(model.parentFile, model.name + ".download")
        val connection = (URL(MODEL_URL).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20000
            readTimeout = 60000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "NaqaaAI-Android")
        }
        try {
            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                throw IllegalStateException("تعذّر تنزيل نموذج إزالة الضبابية (HTTP ${connection.responseCode})")
            }
            BufferedInputStream(connection.inputStream).use { input ->
                FileOutputStream(partial, false).use { output ->
                    val buffer = ByteArray(1024 * 256)
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        output.write(buffer, 0, count)
                    }
                    output.flush()
                    output.fd.sync()
                }
            }
            if (partial.length() < MIN_MODEL_BYTES) {
                throw IllegalStateException("ملف نموذج إزالة الضبابية غير مكتمل")
            }
            if (model.exists()) model.delete()
            if (!partial.renameTo(model)) throw IllegalStateException("تعذّر حفظ نموذج إزالة الضبابية")
        } catch (e: Exception) {
            partial.delete()
            throw e
        } finally {
            connection.disconnect()
        }
    }

    fun restore(source: Bitmap, model: File): Bitmap {
        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
        val output = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
        val pixels = IntArray(TILE * TILE)
        val tensorData = FloatArray(3 * TILE * TILE)
        val canvas = Canvas(output)
        canvas.drawBitmap(source, 0f, 0f, null)

        env.createSession(model.absolutePath, options).use { session ->
            val inputName = session.inputNames.first()
            val outputName = session.outputNames.first()
            var top = 0
            while (top < source.height) {
                var left = 0
                while (left < source.width) {
                    val tile = Bitmap.createBitmap(TILE, TILE, Bitmap.Config.ARGB_8888)
                    val tileCanvas = Canvas(tile)
                    tileCanvas.drawColor(Color.rgb(127, 127, 127))
                    val cropWidth = minOf(TILE, source.width - left)
                    val cropHeight = minOf(TILE, source.height - top)
                    val crop = Bitmap.createBitmap(source, left, top, cropWidth, cropHeight)
                    tileCanvas.drawBitmap(crop, 0f, 0f, paint)
                    tile.getPixels(pixels, 0, TILE, 0, 0, TILE, TILE)
                    for (y in 0 until TILE) {
                        for (x in 0 until TILE) {
                            val idx = y * TILE + x
                            val p = pixels[idx]
                            tensorData[idx] = Color.red(p) / 255f
                            tensorData[TILE * TILE + idx] = Color.green(p) / 255f
                            tensorData[2 * TILE * TILE + idx] = Color.blue(p) / 255f
                        }
                    }
                    OnnxTensor.createTensor(
                        env, FloatBuffer.wrap(tensorData),
                        longArrayOf(1, 3, TILE.toLong(), TILE.toLong())
                    ).use { tensor ->
                        session.run(mapOf(inputName to tensor), setOf(outputName)).use { result ->
                            @Suppress("UNCHECKED_CAST")
                            val values = result.get(outputName).get().value as Array<Array<Array<FloatArray>>>
                            val restored = IntArray(TILE * TILE)
                            for (y in 0 until TILE) {
                                for (x in 0 until TILE) {
                                    val idx = y * TILE + x
                                    restored[idx] = Color.rgb(
                                        (values[0][0][y][x] * 255f).toInt().coerceIn(0, 255),
                                        (values[0][1][y][x] * 255f).toInt().coerceIn(0, 255),
                                        (values[0][2][y][x] * 255f).toInt().coerceIn(0, 255)
                                    )
                                }
                            }
                            val restoredTile = Bitmap.createBitmap(restored, TILE, TILE, Bitmap.Config.ARGB_8888)
                            val valid = Bitmap.createBitmap(restoredTile, 0, 0, cropWidth, cropHeight)
                            canvas.drawBitmap(valid, left.toFloat(), top.toFloat(), paint)
                            valid.recycle()
                            restoredTile.recycle()
                        }
                    }
                    crop.recycle()
                    tile.recycle()
                    left += TILE
                }
                top += TILE
            }
        }
        return output
    }
}
