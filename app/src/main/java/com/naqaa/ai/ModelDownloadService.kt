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
        const val PREFS = "naqaa_model_download"
        private const val CHANNEL_ID = "naqaa_model_downloads"
        private const val NOTIFICATION_ID = 4101
        private const val MODEL_URL =
            "https://huggingface.co/Saimon8420/restoreformer-pp-web/resolve/main/restoreformer_pp_fp16w.onnx"
        private const val EXPECTED_SIZE = 148_000_000L
    }

    private val executor = Executors.newSingleThreadExecutor()
    @Volatile private var isDownloading = false

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val notification = buildNotification("يتم تجهيز تنزيل نموذج الذكاء الاصطناعي", 0, true)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        if (!isDownloading) {
            isDownloading = true
            executor.execute { downloadModel() }
        }
        return START_STICKY
    }

    private fun downloadModel() {
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        val destination = File(filesDir, "restoreformer_pp.onnx")
        val partial = File(filesDir, "restoreformer_pp.onnx.download")
        try {
            if (destination.exists() && destination.length() > 100_000_000L) {
                saveState("completed", destination.length(), destination.length(), null)
                updateNotification("النموذج جاهز للاستخدام دون إنترنت", 100, false)
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
                return
            }

            saveState("downloading", partial.length(), EXPECTED_SIZE, null)
            val connection = (URL(MODEL_URL).openConnection() as HttpURLConnection).apply {
                connectTimeout = 25_000
                readTimeout = 60_000
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", "NaqaaAI-Android")
                if (partial.exists() && partial.length() > 0L) {
                    setRequestProperty("Range", "bytes=${partial.length()}-")
                }
            }
            try {
                val responseCode = connection.responseCode
                if (responseCode != HttpURLConnection.HTTP_OK &&
                    responseCode != HttpURLConnection.HTTP_PARTIAL) {
                    throw IllegalStateException("خادم النموذج أعاد رمز HTTP $responseCode")
                }

                val append = responseCode == HttpURLConnection.HTTP_PARTIAL &&
                    partial.exists() && partial.length() > 0L
                if (!append && partial.exists()) partial.delete()
                val alreadyDownloaded = if (append) partial.length() else 0L
                val contentRange = connection.getHeaderField("Content-Range")
                val rangeTotal = Regex("/(\\d+)$").find(contentRange ?: "")?.groupValues?.get(1)?.toLongOrNull()
                val responseLength = connection.getHeaderFieldLong("Content-Length", -1L)
                val totalBytes = (rangeTotal ?: if (responseLength > 0L) alreadyDownloaded + responseLength else EXPECTED_SIZE)
                    .coerceAtLeast(1L)
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
                                saveState("downloading", downloaded, totalBytes, null)
                                updateNotification(
                                    "تم تنزيل ${formatMb(downloaded)} من ${formatMb(totalBytes)} MB",
                                    ((downloaded * 100L) / totalBytes).toInt().coerceIn(0, 99),
                                    true
                                )
                                lastReported = downloaded
                            }
                        }
                        output.flush()
                        output.fd.sync()
                    }
                }
                if (partial.length() < 100_000_000L) {
                    throw IllegalStateException(
                        "اكتمل الاتصال لكن حجم الملف ${formatMb(partial.length())} MB فقط. أعد المحاولة مع اتصال ثابت."
                    )
                }
                if (destination.exists()) destination.delete()
                if (!partial.renameTo(destination)) {
                    throw IllegalStateException("تعذر نقل النموذج إلى مجلد التطبيق")
                }
                saveState("completed", destination.length(), totalBytes, null)
                updateNotification("اكتمل تنزيل نموذج RestoreFormer++", 100, false)
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            } finally {
                connection.disconnect()
            }
        } catch (e: Exception) {
            saveState("error", partial.length(), EXPECTED_SIZE, e.message ?: "خطأ غير معروف")
            updateNotification("تعذر تنزيل النموذج. افتح نقاء AI للمحاولة مجددًا.", 0, false)
            stopForeground(STOP_FOREGROUND_DETACH)
            stopSelf()
        } finally {
            isDownloading = false
        }
    }

    private fun saveState(state: String, downloaded: Long, total: Long, error: String?) {
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
            .putString("state", state)
            .putLong("downloaded", downloaded)
            .putLong("total", total.coerceAtLeast(1L))
            .putString("error", error ?: "")
            .apply()
    }

    private fun formatMb(bytes: Long): String = String.format(java.util.Locale.US, "%.1f", bytes / 1_000_000.0)

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "تنزيل نموذج RestoreFormer++",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "إظهار تقدم تنزيل نموذج ترميم الوجوه"
                setShowBadge(false)
            }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    private fun buildNotification(text: String, progress: Int, ongoing: Boolean): Notification {
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
        } else {
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
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(NOTIFICATION_ID, buildNotification(text, progress, ongoing))
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        executor.shutdown()
        super.onDestroy()
    }
}
