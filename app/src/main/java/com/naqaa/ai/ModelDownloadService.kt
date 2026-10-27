package com.naqaa.ai

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors

class ModelDownloadService : Service() {
    companion object {
        const val ACTION_START = "com.naqaa.ai.action.START_MODEL_DOWNLOAD"
        const val ACTION_START_CODEFORMER = "com.naqaa.ai.action.START_CODEFORMER_DOWNLOAD"
        const val PREFS = "naqaa_model_download"
        private const val CHANNEL_ID = "naqaa_model_downloads"
        private const val NOTIFICATION_ID = 4101
        private const val NAFNET_URL = "https://huggingface.co/opencv/deblurring_nafnet/resolve/main/deblurring_nafnet_2025may.onnx?download=true"
        private const val CODEFORMER_URL = "https://huggingface.co/bluefoxcreation/Codeformer-ONNX/resolve/main/codeformer.onnx?download=true"
    }

    private val executor = Executors.newSingleThreadExecutor()
    @Volatile private var isDownloading = false

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val codeformer = intent?.action == ACTION_START_CODEFORMER
        val title = if (codeformer) "CodeFormer" else "NAFNet"
        val notification = buildNotification("يتم تجهيز تنزيل نموذج $title", 0, true)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        if (!isDownloading) {
            isDownloading = true
            executor.execute { downloadModel(codeformer) }
        }
        return START_STICKY
    }

    private fun downloadModel(codeformer: Boolean) {
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        val prefix = if (codeformer) "codeformer_" else "nafnet_"
        val title = if (codeformer) "CodeFormer" else "NAFNet"
        val url = if (codeformer) CODEFORMER_URL else NAFNET_URL
        val minimumBytes = if (codeformer) 300_000_000L else 80_000_000L
        val expectedBytes = if (codeformer) 377_000_000L else 92_000_000L
        val fileName = if (codeformer) "codeformer.onnx" else "nafnet_deblur.onnx"
        val destination = File(filesDir, fileName)
        val partial = File(filesDir, "$fileName.download")
        try {
            if (destination.exists() && destination.length() > minimumBytes) {
                saveState(prefix, "completed", destination.length(), destination.length(), null)
                updateNotification("نموذج $title جاهز للاستخدام دون إنترنت", 100, false)
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
                return
            }
            saveState(prefix, "downloading", partial.length(), expectedBytes, null)
            val connection = (URL(url).openConnection() as HttpURLConnection).apply {
                connectTimeout = 25_000
                readTimeout = 90_000
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", "NaqaaAI-Android")
                if (partial.exists() && partial.length() > 0L) setRequestProperty("Range", "bytes=${partial.length()}-")
            }
            try {
                val responseCode = connection.responseCode
                if (responseCode != HttpURLConnection.HTTP_OK && responseCode != HttpURLConnection.HTTP_PARTIAL) {
                    throw IllegalStateException("خادم النموذج أعاد رمز HTTP $responseCode")
                }
                val append = responseCode == HttpURLConnection.HTTP_PARTIAL && partial.exists() && partial.length() > 0L
                if (!append && partial.exists()) partial.delete()
                val alreadyDownloaded = if (append) partial.length() else 0L
                val rangeTotal = Regex("/(\\d+)$").find(connection.getHeaderField("Content-Range") ?: "")?.groupValues?.get(1)?.toLongOrNull()
                val responseLength = connection.getHeaderFieldLong("Content-Length", -1L)
                val totalBytes = (rangeTotal ?: if (responseLength > 0L) alreadyDownloaded + responseLength else expectedBytes).coerceAtLeast(1L)
                var downloaded = alreadyDownloaded
                var lastReported = downloaded
                BufferedInputStream(connection.inputStream).use { input ->
                    FileOutputStream(partial, append).use { output ->
                        val buffer = ByteArray(1024 * 1024)
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            output.write(buffer, 0, count)
                            downloaded += count
                            if (downloaded - lastReported >= 1024 * 1024) {
                                output.flush()
                                saveState(prefix, "downloading", downloaded, totalBytes, null)
                                updateNotification("تنزيل $title: ${formatMb(downloaded)} / ${formatMb(totalBytes)} MB",
                                    ((downloaded * 100L) / totalBytes).toInt().coerceIn(0, 99), true)
                                lastReported = downloaded
                            }
                        }
                        output.flush()
                        output.fd.sync()
                    }
                }
                if (partial.length() < minimumBytes) {
                    throw IllegalStateException("حجم ملف $title غير مكتمل (${formatMb(partial.length())} MB). أعد المحاولة.")
                }
                if (destination.exists()) destination.delete()
                if (!partial.renameTo(destination)) throw IllegalStateException("تعذر حفظ ملف النموذج على الجهاز")
                saveState(prefix, "completed", destination.length(), totalBytes, null)
                updateNotification("اكتمل تنزيل نموذج $title", 100, false)
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            } finally {
                connection.disconnect()
            }
        } catch (e: Exception) {
            saveState(prefix, "error", partial.length(), expectedBytes, e.message ?: "خطأ غير معروف")
            updateNotification("تعذر تنزيل $title. افتح نقاء AI للمحاولة مجددًا.", 0, false)
            stopForeground(STOP_FOREGROUND_DETACH)
            stopSelf()
        } finally {
            isDownloading = false
        }
    }

    private fun saveState(prefix: String, state: String, downloaded: Long, total: Long, error: String?) {
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
            .putString(prefix + "state", state)
            .putLong(prefix + "downloaded", downloaded)
            .putLong(prefix + "total", total.coerceAtLeast(1L))
            .putString(prefix + "error", error ?: "")
            .apply()
    }

    private fun formatMb(bytes: Long) = String.format(java.util.Locale.US, "%.1f", bytes / 1_000_000.0)

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(CHANNEL_ID, "تنزيل نماذج نقاء AI", NotificationManager.IMPORTANCE_LOW).apply {
                description = "إظهار تقدم تنزيل نماذج الذكاء الاصطناعي"
                setShowBadge(false)
            }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    private fun buildNotification(text: String, progress: Int, ongoing: Boolean): Notification {
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) Notification.Builder(this, CHANNEL_ID)
        else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
        }
        builder.setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("نقاء AI — تنزيل النموذج")
            .setContentText(text)
            .setOnlyAlertOnce(true)
            .setOngoing(ongoing)
            .setProgress(100, progress.coerceIn(0, 100), ongoing && progress == 0)
        return builder.build()
    }

    private fun updateNotification(text: String, progress: Int, ongoing: Boolean) {
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, buildNotification(text, progress, ongoing))
    }

    override fun onBind(intent: Intent?): IBinder? = null
    override fun onDestroy() {
        executor.shutdown()
        super.onDestroy()
    }
}
