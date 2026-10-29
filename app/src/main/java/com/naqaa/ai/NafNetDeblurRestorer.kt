package com.naqaa.ai

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
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
 * Overlapping tiles avoid zero-sized intermediate tensors in the encoder.
 */
internal object NafNetDeblurRestorer {
    private const val MODEL_URL =
        "https://huggingface.co/opencv/deblurring_nafnet/resolve/main/deblurring_nafnet_2025may.onnx?download=true"
    private const val MIN_MODEL_BYTES = 80_000_000L
    private const val TILE = 512
    private const val STEP = 448

    fun ensureModel(filesDir: File, onProgress: (downloaded: Long, total: Long) -> Unit = { _, _ -> }): File {
        val destination = File(filesDir, "nafnet_deblur.onnx")
        if (destination.isFile && destination.length() >= MIN_MODEL_BYTES) {
            onProgress(destination.length(), destination.length())
            return destination
        }

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
            val total = connection.contentLengthLong
            var downloaded = 0L
            var lastReported = 0L
            connection.inputStream.use { input ->
                FileOutputStream(partial).use { output ->
                    val buffer = ByteArray(256 * 1024)
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        output.write(buffer, 0, count)
                        downloaded += count
                        if (downloaded - lastReported >= 512 * 1024 || (total > 0 && downloaded >= total)) {
                            onProgress(downloaded, total)
                            lastReported = downloaded
                        }
                    }
                }
            }
            if (partial.length() < MIN_MODEL_BYTES) {
                throw IllegalStateException("ملف النموذج غير مكتمل؛ أعد المحاولة باتصال ثابت")
            }
            if (destination.exists()) destination.delete()
            if (!partial.renameTo(destination)) {
                throw IllegalStateException("تعذر حفظ ملف النموذج على الجهاز")
            }
            onProgress(destination.length(), if (total > 0) total else destination.length())
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
            val outputName = session.outputNames.firstOrNull()
                ?: throw IllegalStateException("النموذج لا يحتوي على خرج")
            val inputInfo = session.inputInfo[inputName]?.info as? ai.onnxruntime.TensorInfo
            val outputInfo = session.outputInfo[outputName]?.info as? ai.onnxruntime.TensorInfo
            Log.i("NaqaaNAFNet", "Model input=$inputName shape=${inputInfo?.shape?.contentToString()} type=${inputInfo?.type}; output=$outputName shape=${outputInfo?.shape?.contentToString()} type=${outputInfo?.type}")
            var comparedChannels = 0L
            var absDiffSum = 0L
            var outputMin = Float.POSITIVE_INFINITY
            var outputMax = Float.NEGATIVE_INFINITY
            var outputNonFinite = 0L
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
                            if (values.size != 1 || values[0].size != 3 ||
                                values[0][0].size != TILE || values[0][0][0].size != TILE ||
                                values[0][1].size != TILE || values[0][2].size != TILE ||
                                values[0].any { plane -> plane.any { row -> row.size != TILE } }) {
                                throw IllegalStateException("أبعاد خرج النموذج غير متوافقة: المتوقع [1,3,$TILE,$TILE]")
                            }

                            val localLeft = if (left == 0) 0 else 32
                            val localTop = if (top == 0) 0 else 32
                            val localRight = if (left + TILE >= width) TILE else TILE - 32
                            val localBottom = if (top + TILE >= height) TILE else TILE - 32
                            for (y in localTop until localBottom) {
                                val gy = top + y
                                if (gy !in 0 until height) continue
                                for (x in localLeft until localRight) {
                                    val gx = left + x
                                    if (gx !in 0 until width) continue
                                    val rawR = values[0][0][y][x]
                                    val rawG = values[0][1][y][x]
                                    val rawB = values[0][2][y][x]
                                    for (raw in floatArrayOf(rawR, rawG, rawB)) {
                                        if (!raw.isFinite()) {
                                            outputNonFinite++
                                        } else {
                                            outputMin = minOf(outputMin, raw)
                                            outputMax = maxOf(outputMax, raw)
                                        }
                                    }
                                    val sourceColor = source.getPixel(gx, gy)
                                    absDiffSum += kotlin.math.abs((rawR * 255f).toInt().coerceIn(0, 255) - Color.red(sourceColor))
                                    absDiffSum += kotlin.math.abs((rawG * 255f).toInt().coerceIn(0, 255) - Color.green(sourceColor))
                                    absDiffSum += kotlin.math.abs((rawB * 255f).toInt().coerceIn(0, 255) - Color.blue(sourceColor))
                                    comparedChannels += 3
                                    val r = (rawR * 255f).toInt().coerceIn(0, 255)
                                    val g = (rawG * 255f).toInt().coerceIn(0, 255)
                                    val b = (rawB * 255f).toInt().coerceIn(0, 255)
                                    outputPixels[gy * width + gx] = Color.rgb(r, g, b)
                                }
                            }
                        }
                    }
                }
            }
            val meanAbsoluteDifference = if (comparedChannels == 0L) 0.0 else absDiffSum.toDouble() / comparedChannels
            Log.i("NaqaaNAFNet", "Output stats: rawMin=$outputMin rawMax=$outputMax nonFinite=$outputNonFinite; meanAbsDiffFromInput=$meanAbsoluteDifference/255 across $comparedChannels channels")
        }
        return Bitmap.createBitmap(outputPixels, width, height, Bitmap.Config.ARGB_8888)
    }

    /**
     * Distribute tile starts evenly across the image. The old fixed-step loop
     * could add a regular tile and then a nearly identical end-aligned tile
     * (e.g. x=448 and x=488 for a 1000px image), causing excessive overwrites.
     */
    private fun tileStarts(size: Int): List<Int> {
        if (size <= TILE) return listOf(0)
        val last = size - TILE
        val tileCount = kotlin.math.ceil(last.toDouble() / STEP).toInt() + 1
        if (tileCount <= 2) return listOf(0, last)
        return (0 until tileCount).map { index ->
            (last.toLong() * index / (tileCount - 1)).toInt()
        }.distinct()
    }
}
