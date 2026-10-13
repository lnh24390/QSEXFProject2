package com.neondrift.game;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowManager;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.FrameLayout;
import android.widget.TextView;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Collections;

/** Offline WebView shell. No network permission, native JavaScript bridge or file access. */
public final class MainActivity extends Activity {
    private static final String HOST = "appassets.androidplatform.net";
    private static final String HOME = "https://" + HOST + "/assets/index.html";
    private WebView webView;
    private boolean foreground;
    private boolean backPending;
    private AlertDialog exitDialog;

    @Override
    @SuppressLint("SetJavaScriptEnabled")
    public void onCreate(Bundle savedState) {
        super.onCreate(savedState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.rgb(9, 11, 19));
        root.setOnApplyWindowInsetsListener((view, insets) -> {
            if (Build.VERSION.SDK_INT >= 30) {
                android.graphics.Insets safe = insets.getInsets(
                    WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                view.setPadding(safe.left, safe.top, safe.right, safe.bottom);
            } else {
                view.setPadding(insets.getSystemWindowInsetLeft(), insets.getSystemWindowInsetTop(),
                    insets.getSystemWindowInsetRight(), insets.getSystemWindowInsetBottom());
            }
            return Build.VERSION.SDK_INT >= 30 ? WindowInsets.CONSUMED : insets.consumeSystemWindowInsets();
        });
        setContentView(root);
        try {
            webView = new WebView(this);
        } catch (RuntimeException unavailable) {
            TextView message = new TextView(this);
            message.setText("Android System WebView를 활성화하거나 업데이트한 뒤 다시 실행해주세요.");
            message.setTextColor(Color.WHITE);
            message.setPadding(32, 80, 32, 32);
            root.addView(message);
            return;
        }
        webView.setBackgroundColor(Color.rgb(9, 11, 19));
        webView.setOverScrollMode(View.OVER_SCROLL_NEVER);
        webView.setHorizontalScrollBarEnabled(false);
        webView.setVerticalScrollBarEnabled(false);
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(false);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        settings.setMediaPlaybackRequiresUserGesture(true);
        settings.setSupportZoom(false);
        settings.setTextZoom(100);
        webView.setWebViewClient(new LocalClient());
        root.addView(webView, new FrameLayout.LayoutParams(-1, -1));
        root.requestApplyInsets();
        // Restore scores through WebView localStorage; an interrupted run starts fresh.
        webView.loadUrl(HOME);
        if (Build.VERSION.SDK_INT >= 33) {
            getOnBackInvokedDispatcher().registerOnBackInvokedCallback(
                android.window.OnBackInvokedDispatcher.PRIORITY_DEFAULT, this::handleBack);
        }
    }

    private final class LocalClient extends WebViewClient {
        @Override public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
            return !HOME.equals(request.getUrl().toString());
        }
        @Override public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
            Uri uri = request.getUrl();
            String path = uri.getPath();
            if (!"GET".equals(request.getMethod()) || !"https".equals(uri.getScheme()) ||
                !HOST.equals(uri.getHost()) || path == null || !path.startsWith("/assets/") ||
                path.contains("..") || path.contains("\\")) return notFound();
            String asset = path.substring("/assets/".length());
            try {
                String mime = asset.endsWith(".html") ? "text/html" :
                    asset.endsWith(".css") ? "text/css" :
                    asset.endsWith(".js") ? "application/javascript" :
                    asset.endsWith(".png") ? "image/png" :
                    asset.endsWith(".jpg") || asset.endsWith(".jpeg") ? "image/jpeg" :
                    asset.endsWith(".webp") ? "image/webp" : "application/octet-stream";
                return new WebResourceResponse(mime, "UTF-8", 200, "OK",
                    Collections.singletonMap("Cache-Control", "no-cache"), getAssets().open("web/" + asset));
            } catch (IOException missing) { return notFound(); }
        }
    }
    private static WebResourceResponse notFound() {
        return new WebResourceResponse("text/plain", "UTF-8", 404, "Not Found",
            Collections.emptyMap(), new ByteArrayInputStream(new byte[0]));
    }

    @Override public void onBackPressed() { handleBack(); }
    private void handleBack() {
        if (backPending || (exitDialog != null && exitDialog.isShowing())) return;
        if (webView == null) { finish(); return; }
        backPending = true;
        webView.evaluateJavascript("Boolean(window.neonAndroidBack && window.neonAndroidBack())", result -> {
            backPending = false;
            if (!"true".equals(result) && !isFinishing() && !isDestroyed()) {
                exitDialog = new AlertDialog.Builder(this)
                    .setTitle("게임을 종료할까요?")
                    .setMessage("최고 기록은 이 기기에 저장됩니다.")
                    .setNegativeButton("취소", null)
                    .setPositiveButton("종료", (dialog, which) -> finish()).show();
            }
        });
    }
    @Override protected void onPause() {
        foreground = false;
        if (webView != null) {
            webView.evaluateJavascript("window.neonAndroidPause && window.neonAndroidPause()", result -> {
                if (!foreground && webView != null) { webView.onPause(); webView.pauseTimers(); }
            });
        }
        super.onPause();
    }
    @Override protected void onResume() {
        super.onResume(); foreground = true;
        if (webView != null) { webView.resumeTimers(); webView.onResume(); }
    }
    @Override protected void onDestroy() {
        if (webView != null) { webView.stopLoading(); webView.destroy(); webView = null; }
        super.onDestroy();
    }
}
