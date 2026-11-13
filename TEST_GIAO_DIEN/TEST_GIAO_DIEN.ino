#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include <time.h>

// =======================
// MAN HINH 1.8" TFT SPI 128x160 v1.1 (ST7735)
// =======================
// Noi day ESP32 (VSPI chuan):
//   VCC       -> 3V3 hoac 5V
//   GND       -> GND
//   SCL (SCK) -> GPIO 18
//   SDA (MOSI)-> GPIO 23
//   RES (RST) -> GPIO 4
//   DC  (A0)  -> GPIO 2
//   CS        -> GPIO 5
//   BL  (LED) -> 3V3
#define TFT_CS    5
#define TFT_RST   4
#define TFT_DC    2

// =======================
// COI BAO DONG (BUZZER) & LED DO
// =======================
// Luu y: GPIO 19 bi thu vien SPI cua man TFT chiem lam chan MISO!
// Vi vay dung GPIO 21 (D21) va GPIO 22 (D22) hoan toan doc lap:
//   Coi (Buzzer): chan (+) -> GPIO 21 (D21), chan (-) -> GND
//   LED do      : chan (+) -> GPIO 22 (D22), chan (-) -> GND
#define BUZZER_PIN   21
#define LED_RED_PIN  22

void setAlarmHardware(bool isFall) {
  digitalWrite(BUZZER_PIN,  isFall ? HIGH : LOW);
  digitalWrite(LED_RED_PIN, isFall ? HIGH : LOW);
}

Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_RST);

// =======================
// WEB SERVER + MEMORY
// =======================
WebServer server(80);
Preferences preferences;

// =======================
// WIFI AP CAI DAT
// =======================
const char* ap_ssid = "SET_UP";
const char* ap_password = "12345678";

// =======================
// NTP TIME - VIET NAM UTC+7
// =======================
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 7 * 3600;
const int daylightOffset_sec = 0;

// =======================
// BIEN WIFI
// =======================
String savedSSID = "";
String savedPASS = "";
bool setupMode = false;

// =======================
// CAMERA URL
// =======================
// Neu dung Flask AI:
// String cameraURL = "http://192.168.1.15:5000/video_feed";

// Neu dung ESP32-CAM:
// String cameraURL = "http://192.168.1.88/stream";

// Luong video AI tu Raspberry Pi (doi IP neu Pi doi IP)
String cameraURL = "http://192.168.1.103:5000/video_feed";

// =======================
// BIEN HE THONG
// =======================
String aiState = "NORMAL";
String alertText = "Hệ thống đang theo dõi an toàn";
String wifiStatus = "OFF";
String cameraStatus = "Chưa cấu hình";
String dfStatus = "Chưa kết nối";
String simStatus = "Chưa kết nối";
String smsStatus = "Chưa gửi";
String callStatus = "Chưa gọi";
String systemIP = "";
String lastFallTime = "Chưa có";
float confidence = 0.00;

// =======================
// NHAT KY HE THONG
// =======================
#define MAX_LOGS 10

String detectionLogs[MAX_LOGS];
int logCount = 0;
String lastLoggedState = "";

// =======================
// DANH SACH THOI GIAN TE NGA
// =======================
#define MAX_FALL_TIMES 10

String fallTimeLogs[MAX_FALL_TIMES];
int fallTimeCount = 0;

// =======================
// HIEN THI MAN HINH 1.8" TFT ST7735 (160x128 ngang)
// =======================
void showOLED(String line1, String line2, String line3) {
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextWrap(false);

  // Thanh tieu de tren cung
  uint16_t headerColor = ST77XX_BLUE;
  uint16_t stateColor  = ST77XX_GREEN;
  if (line1.indexOf("FALL") >= 0 && line1.indexOf("SETUP") < 0 && line1.indexOf("SYS") < 0) {
    headerColor = ST77XX_RED;
    stateColor  = ST77XX_RED;
  } else if (line1.indexOf("WARN") >= 0) {
    headerColor = ST77XX_ORANGE;
    stateColor  = ST77XX_YELLOW;
  }

  tft.fillRect(0, 0, 160, 22, headerColor);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(8, 7);
  tft.println("SM-LAB FALL DETECTION");

  // Dong 1: Trang thai chinh (co chu lon, toi da 12 ky tu tren man 160px)
  tft.setTextColor(stateColor);
  tft.setTextSize(2);
  tft.setCursor(6, 30);
  tft.println(line1);

  // Duong ke ngang phan cach
  tft.drawFastHLine(6, 54, 148, 0x7BEF);

  // Dong 2: Thong tin chi tiet
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(6, 64);
  tft.println(line2);

  // Dong 3: Dia chi IP / Huong dan
  tft.setTextColor(ST77XX_CYAN);
  tft.setTextSize(1);
  tft.setCursor(6, 84);
  tft.println(line3);

  // Dong 4: Thoi gian nga gan nhat (bo dau ngoac vuong de vua khit 1 dong 160px)
  String fallText = lastFallTime;
  if (fallText == "Chưa có") {
    fallText = "None";
  } else {
    fallText.replace("[", "");
    fallText.replace("]", "");
  }
  tft.setTextColor(ST77XX_YELLOW);
  tft.setTextSize(1);
  tft.setCursor(5, 106);
  tft.println("Fall: " + fallText);
}

// =======================
// LAY THOI GIAN THUC
// =======================
String getTimeNow() {
  struct tm timeinfo;

  if (!getLocalTime(&timeinfo)) {
    return "[" + String(millis() / 1000) + "s]";
  }

  char timeString[30];
  strftime(timeString, sizeof(timeString), "[%d/%m/%Y %H:%M:%S]", &timeinfo);

  return String(timeString);
}

// =======================
// THEM NHAT KY HE THONG
// =======================
void addDetectionLog(String message) {
  String timeText = getTimeNow();
  String newLog = timeText + " " + message;

  for (int i = MAX_LOGS - 1; i > 0; i--) {
    detectionLogs[i] = detectionLogs[i - 1];
  }

  detectionLogs[0] = newLog;

  if (logCount < MAX_LOGS) {
    logCount++;
  }
}

// =======================
// THEM THOI GIAN TE NGA
// =======================
void addFallTimeLog(String timeText) {
  for (int i = MAX_FALL_TIMES - 1; i > 0; i--) {
    fallTimeLogs[i] = fallTimeLogs[i - 1];
  }

  fallTimeLogs[0] = timeText;

  if (fallTimeCount < MAX_FALL_TIMES) {
    fallTimeCount++;
  }
}

// =======================
// NHAT KY HE THONG SANG JSON
// =======================
String getLogsJSON() {
  String logs = "[";

  for (int i = 0; i < logCount; i++) {
    String item = detectionLogs[i];
    item.replace("\"", "'");

    logs += "\"";
    logs += item;
    logs += "\"";

    if (i < logCount - 1) {
      logs += ",";
    }
  }

  logs += "]";
  return logs;
}

// =======================
// DANH SACH THOI GIAN TE NGA SANG JSON
// =======================
String getFallTimesJSON() {
  String json = "[";

  for (int i = 0; i < fallTimeCount; i++) {
    String item = fallTimeLogs[i];
    item.replace("\"", "'");

    json += "\"";
    json += item;
    json += "\"";

    if (i < fallTimeCount - 1) {
      json += ",";
    }
  }

  json += "]";
  return json;
}

// =======================
// DOC WIFI DA LUU
// =======================
void loadWiFi() {
  preferences.begin("wifi", true);
  savedSSID = preferences.getString("ssid", "Chi Tam");
  savedPASS = preferences.getString("pass", "0909747517");
  preferences.end();
}

// =======================
// LUU WIFI
// =======================
void saveWiFi(String ssid, String pass) {
  preferences.begin("wifi", false);
  preferences.putString("ssid", ssid);
  preferences.putString("pass", pass);
  preferences.end();
}

// =======================
// XOA WIFI
// =======================
void clearWiFi() {
  preferences.begin("wifi", false);
  preferences.clear();
  preferences.end();
}

// =======================
// TRANG CAI DAT WIFI
// =======================
String wifiSetupPage() {
  String html = R"rawliteral(
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cài đặt WiFi</title>

<style>
body {
  margin: 0;
  background: linear-gradient(135deg, #0f172a, #1d4ed8);
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: Arial, sans-serif;
}

.box {
  width: 92%;
  max-width: 430px;
  background: white;
  border-radius: 18px;
  padding: 24px;
  box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}

h1 {
  text-align: center;
  color: #0f172a;
}

p {
  text-align: center;
  color: #475569;
}

label {
  display: block;
  margin-top: 14px;
  font-weight: bold;
}

input {
  width: 100%;
  padding: 13px;
  margin-top: 6px;
  border: 1px solid #cbd5e1;
  border-radius: 10px;
  font-size: 16px;
  box-sizing: border-box;
}

button {
  width: 100%;
  margin-top: 20px;
  padding: 14px;
  border: none;
  border-radius: 12px;
  background: #2563eb;
  color: white;
  font-size: 17px;
  font-weight: bold;
}

.note {
  margin-top: 15px;
  background: #eff6ff;
  color: #1e40af;
  padding: 12px;
  border-radius: 10px;
  line-height: 1.6;
}
</style>
</head>

<body>
<div class="box">
  <h1>Cài đặt WiFi</h1>
  <p>Nhập WiFi để hệ thống kết nối mạng.</p>

  <form action="/savewifi" method="POST">
    <label>Tên WiFi</label>
    <input name="ssid" placeholder="Nhập tên WiFi" required>

    <label>Mật khẩu WiFi</label>
    <input name="password" type="password" placeholder="Nhập mật khẩu">

    <button type="submit">Lưu và kết nối</button>
  </form>

  <div class="note">
    WiFi cài đặt: FALL_SETUP<br>
    Mật khẩu: 12345678<br>
    Địa chỉ: 192.168.4.1
  </div>
</div>
</body>
</html>
)rawliteral";

  return html;
}

// =======================
// TRANG HE THONG
// =======================
String systemPage() {
  String html = R"rawliteral(
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hệ thống cảnh báo té ngã</title>

<style>
* {
  box-sizing: border-box;
  font-family: Arial, sans-serif;
}

body {
  margin: 0;
  background: #eef2f7;
  color: #111827;
}

.header {
  background: linear-gradient(135deg, #0f172a, #1d4ed8);
  color: white;
  padding: 20px 12px;
  text-align: center;
}

.header h2 {
  margin: 0;
  font-size: 22px;
}

.header p {
  margin: 8px 0 0;
  color: #dbeafe;
}

.container {
  padding: 16px;
  display: grid;
  grid-template-columns: 1.3fr 1fr;
  gap: 16px;
}

.card {
  background: white;
  padding: 18px;
  border-radius: 16px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.08);
}

.card h3 {
  margin-top: 0;
  color: #0f172a;
}

.camera-box {
  width: 100%;
  height: 360px;
  background: #0f172a;
  color: #e5e7eb;
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.camera-box img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.state {
  font-size: 34px;
  text-align: center;
  font-weight: bold;
  padding: 30px;
  border-radius: 16px;
  margin-bottom: 14px;
}

.normal {
  background: #dcfce7;
  color: #166534;
}

.warning {
  background: #fef9c3;
  color: #854d0e;
}

.fall {
  background: #fee2e2;
  color: #991b1b;
}

.row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 11px 0;
  border-bottom: 1px solid #e5e7eb;
}

.label {
  font-weight: bold;
  color: #475569;
}

.value {
  text-align: right;
  font-weight: 500;
}

.ok {
  color: #16a34a;
  font-weight: bold;
}

.warn {
  color: #ca8a04;
  font-weight: bold;
}

.danger {
  color: #dc2626;
  font-weight: bold;
}

button {
  width: 100%;
  margin-top: 10px;
  padding: 14px;
  border: none;
  border-radius: 12px;
  color: white;
  font-weight: bold;
  font-size: 16px;
  cursor: pointer;
}

.btn-normal {
  background: #22c55e;
}

.btn-warning {
  background: #facc15;
  color: #111827;
}

.btn-fall {
  background: #ef4444;
}

.btn-sms {
  background: #3b82f6;
}

.btn-call {
  background: #7c3aed;
}

.btn-reset {
  background: #64748b;
}

.btn-wifi {
  background: #0f172a;
}

.full {
  grid-column: 1 / -1;
}

.log {
  background: #0f172a;
  color: #e5e7eb;
  border-radius: 12px;
  padding: 14px;
  min-height: 130px;
  max-height: 260px;
  overflow-y: auto;
  line-height: 1.7;
  font-size: 15px;
}

.fall-list {
  background: #fff7ed;
  color: #7c2d12;
  border: 1px solid #fed7aa;
  border-radius: 12px;
  padding: 14px;
  min-height: 100px;
  max-height: 240px;
  overflow-y: auto;
  line-height: 1.8;
  font-size: 15px;
  font-weight: bold;
}

.footer {
  text-align: center;
  padding: 12px;
  color: #64748b;
  font-size: 14px;
}

@media(max-width: 850px) {
  .container {
    grid-template-columns: 1fr;
  }

  .camera-box {
    height: 240px;
  }

  .header h2 {
    font-size: 19px;
  }
}
</style>
</head>

<body>

<div class="header">
  <h2>HỆ THỐNG THEO DÕI VÀ CẢNH BÁO TÉ NGÃ NGƯỜI GIÀ</h2>
  <p>ESP32 WebServer • Camera • AI • DFPlayer • SIM SMS/Call</p>
</div>

<div class="container">

  <div class="card">
    <h3>Camera giám sát</h3>
    <div class="camera-box" id="cameraBox">
      Đang tải camera...
    </div>
  </div>

  <div class="card">
    <h3>Trạng thái AI</h3>

    <div class="state normal" id="stateBox">
      NORMAL
    </div>

    <div class="row">
      <span class="label">Độ tin cậy:</span>
      <span id="confidence" class="value">0.00</span>
    </div>

    <div class="row">
      <span class="label">Cảnh báo:</span>
      <span id="alertText" class="value">---</span>
    </div>

    <div class="row">
      <span class="label">Lần té ngã gần nhất:</span>
      <span id="lastFallTime" class="value">Chưa có</span>
    </div>

    <div class="row">
      <span class="label">Thời gian hoạt động:</span>
      <span id="uptime" class="value">---</span>
    </div>
  </div>

  <div class="card">
    <h3>Trạng thái thiết bị</h3>

    <div class="row">
      <span class="label">WiFi:</span>
      <span id="wifi" class="ok">---</span>
    </div>

    <div class="row">
      <span class="label">IP:</span>
      <span id="ip" class="value">---</span>
    </div>

    <div class="row">
      <span class="label">Camera:</span>
      <span id="cameraStatus" class="warn">---</span>
    </div>

    <div class="row">
      <span class="label">DFPlayer:</span>
      <span id="dfStatus" class="warn">---</span>
    </div>

    <div class="row">
      <span class="label">SIM:</span>
      <span id="simStatus" class="warn">---</span>
    </div>

    <div class="row">
      <span class="label">SMS:</span>
      <span id="smsStatus" class="value">---</span>
    </div>

    <div class="row">
      <span class="label">Cuộc gọi:</span>
      <span id="callStatus" class="value">---</span>
    </div>
  </div>

  <div class="card">
    <h3>Điều khiển thử nghiệm</h3>

    <button class="btn-normal" onclick="sendCmd('/set?state=normal')">
      NORMAL - Bình thường
    </button>

    <button class="btn-warning" onclick="sendCmd('/set?state=warning')">
      WARNING - Tư thế bất thường
    </button>

    <button class="btn-fall" onclick="sendCmd('/set?state=fall')">
      FALL - Té ngã
    </button>

    <button class="btn-sms" onclick="sendCmd('/sms')">
      Test gửi SMS
    </button>

    <button class="btn-call" onclick="sendCmd('/call')">
      Test gọi điện
    </button>

    <button class="btn-reset" onclick="sendCmd('/reset')">
      Reset trạng thái
    </button>

    <button class="btn-wifi" onclick="clearWiFi()">
      Cài đặt lại WiFi
    </button>
  </div>

  <div class="card full">
    <h3>Danh sách thời gian té ngã</h3>
    <div id="fallTimes" class="fall-list">
      Chưa có thời gian té ngã.
    </div>
  </div>

  <div class="card full">
    <h3>Nhật ký hệ thống</h3>
    <div id="log" class="log">
      Chưa có nhật ký phát hiện.
    </div>
  </div>

</div>

<div class="footer">
  Đồ án: Hệ thống phát hiện và cảnh báo té ngã cho người già
</div>

<script>
function updateData() {
  fetch('/data')
    .then(res => res.json())
    .then(data => {
      let stateBox = document.getElementById("stateBox");

      stateBox.innerHTML = data.state;
      stateBox.className = "state " + data.stateClass;

      document.getElementById("confidence").innerHTML = data.confidence;
      document.getElementById("alertText").innerHTML = data.alertText;
      document.getElementById("wifi").innerHTML = data.wifi;
      document.getElementById("ip").innerHTML = data.ip;
      document.getElementById("uptime").innerHTML = data.uptime;
      document.getElementById("cameraStatus").innerHTML = data.cameraStatus;
      document.getElementById("dfStatus").innerHTML = data.dfStatus;
      document.getElementById("simStatus").innerHTML = data.simStatus;
      document.getElementById("smsStatus").innerHTML = data.smsStatus;
      document.getElementById("callStatus").innerHTML = data.callStatus;
      document.getElementById("lastFallTime").innerHTML = data.lastFallTime;

      let cameraBox = document.getElementById("cameraBox");

      if (data.cameraURL.length > 0) {
        cameraBox.innerHTML = '<img src="' + data.cameraURL + '" alt="Camera">';
      } else {
        cameraBox.innerHTML = 'Chưa cấu hình đường dẫn camera';
      }

      let fallTimeText = "";

      if (data.fallTimes.length > 0) {
        for (let i = 0; i < data.fallTimes.length; i++) {
          fallTimeText += (i + 1) + ". " + data.fallTimes[i] + "<br>";
        }
      } else {
        fallTimeText = "Chưa có thời gian té ngã.";
      }

      document.getElementById("fallTimes").innerHTML = fallTimeText;

      let logText = "";

      if (data.logs.length > 0) {
        for (let i = 0; i < data.logs.length; i++) {
          logText += data.logs[i] + "<br>";
        }
      } else {
        logText = "Chưa có nhật ký phát hiện.";
      }

      document.getElementById("log").innerHTML = logText;
    });
}

function sendCmd(url) {
  fetch(url)
    .then(() => updateData());
}

function clearWiFi() {
  if (confirm("Xóa WiFi đã lưu và cài đặt lại?")) {
    fetch('/clearwifi')
      .then(res => res.text())
      .then(text => alert(text));
  }
}

setInterval(updateData, 1000);
updateData();
</script>

</body>
</html>
)rawliteral";

  return html;
}

// =======================
// ROUTE ROOT
// =======================
void handleRoot() {
  if (setupMode) {
    server.send(200, "text/html; charset=utf-8", wifiSetupPage());
  } else {
    server.send(200, "text/html; charset=utf-8", systemPage());
  }
}

// =======================
// ROUTE SAVE WIFI
// =======================
void handleSaveWiFi() {
  if (!server.hasArg("ssid")) {
    server.send(400, "text/plain; charset=utf-8", "Thiếu tên WiFi");
    return;
  }

  String ssid = server.arg("ssid");
  String pass = server.arg("password");

  saveWiFi(ssid, pass);

  String html = R"rawliteral(
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Đã lưu WiFi</title>
</head>
<body style="font-family:Arial;text-align:center;padding:40px;">
<h2>Đã lưu WiFi</h2>
<p>ESP32 sẽ khởi động lại và kết nối WiFi vừa nhập.</p>
<p>Sau đó xem OLED hoặc Serial Monitor để biết IP hệ thống.</p>
</body>
</html>
)rawliteral";

  server.send(200, "text/html; charset=utf-8", html);

  delay(2000);
  ESP.restart();
}

// =======================
// ROUTE DATA
// =======================
void handleData() {
  String ip = "";

  if (setupMode) {
    ip = WiFi.softAPIP().toString();
  } else {
    ip = WiFi.localIP().toString();
  }

  if (cameraURL.length() > 0) {
    cameraStatus = "OK";
  } else {
    cameraStatus = "Chưa cấu hình";
  }

  String stateClass = "normal";

  if (aiState == "WARNING") {
    stateClass = "warning";
  }

  if (aiState == "FALL") {
    stateClass = "fall";
  }

  String json = "{";
  json += "\"state\":\"" + aiState + "\",";
  json += "\"stateClass\":\"" + stateClass + "\",";
  json += "\"confidence\":\"" + String(confidence, 2) + "\",";
  json += "\"alertText\":\"" + alertText + "\",";
  json += "\"wifi\":\"" + wifiStatus + "\",";
  json += "\"ip\":\"" + ip + "\",";
  json += "\"uptime\":\"" + String(millis() / 1000) + "s\",";
  json += "\"cameraStatus\":\"" + cameraStatus + "\",";
  json += "\"cameraURL\":\"" + cameraURL + "\",";
  json += "\"dfStatus\":\"" + dfStatus + "\",";
  json += "\"simStatus\":\"" + simStatus + "\",";
  json += "\"smsStatus\":\"" + smsStatus + "\",";
  json += "\"callStatus\":\"" + callStatus + "\",";
  json += "\"lastFallTime\":\"" + lastFallTime + "\",";
  json += "\"fallTimes\":" + getFallTimesJSON() + ",";
  json += "\"logs\":" + getLogsJSON();
  json += "}";

  server.send(200, "application/json; charset=utf-8", json);
}

// =======================
// ROUTE SET STATE
// =======================
void handleSetState() {
  if (!server.hasArg("state")) {
    server.send(400, "text/plain; charset=utf-8", "Thiếu trạng thái");
    return;
  }

  String state = server.arg("state");

  if (state == "normal") {
    aiState = "NORMAL";
    confidence = 0.95;
    alertText = "Hệ thống đang theo dõi an toàn";
    setAlarmHardware(false);

    if (lastLoggedState != "NORMAL") {
      addDetectionLog("Trạng thái bình thường - hệ thống an toàn");
      lastLoggedState = "NORMAL";
    }

    showOLED("SAFE / OK", "State: NORMAL", "IP: " + systemIP);
    server.send(200, "text/plain; charset=utf-8", "Đã chuyển NORMAL");
  }

  else if (state == "warning") {
    aiState = "WARNING";
    confidence = 0.88;
    alertText = "Phát hiện tư thế bất thường";
    setAlarmHardware(false);

    if (lastLoggedState != "WARNING") {
      addDetectionLog("Phát hiện tư thế bất thường - cần theo dõi");
      lastLoggedState = "WARNING";
    }

    showOLED("WARNING!", "Tu the bat thuong", "Check now");
    server.send(200, "text/plain; charset=utf-8", "Đã chuyển WARNING");
  }

  else if (state == "fall") {
    aiState = "FALL";
    confidence = 0.93;
    alertText = "Cảnh báo! Phát hiện người già bị té ngã";
    setAlarmHardware(true);

    String fallTime = getTimeNow();
    lastFallTime = fallTime;

    if (lastLoggedState != "FALL") {
      addDetectionLog("CẢNH BÁO TÉ NGÃ - cần kiểm tra ngay");
      addFallTimeLog(fallTime);
      lastLoggedState = "FALL";
    }

    showOLED("FALL ALERT!", "Phat hien te nga", "Kiem tra ngay");
    server.send(200, "text/plain; charset=utf-8", "Đã chuyển FALL");
  }

  else {
    server.send(400, "text/plain; charset=utf-8", "Trạng thái không hợp lệ");
  }
}

// =======================
// ROUTE TEST SMS
// =======================
void handleSMS() {
  smsStatus = "Đã gửi lệnh test SMS";
  addDetectionLog("Đã gửi cảnh báo SMS đến người thân");
  server.send(200, "text/plain; charset=utf-8", "Đã test SMS");
}

// =======================
// ROUTE TEST CALL
// =======================
void handleCall() {
  callStatus = "Đã gửi lệnh test gọi điện";
  addDetectionLog("Đã thực hiện cuộc gọi khẩn cấp");
  server.send(200, "text/plain; charset=utf-8", "Đã test CALL");
}

// =======================
// ROUTE RESET
// =======================
void handleReset() {
  aiState = "NORMAL";
  confidence = 0.00;
  alertText = "Hệ thống đã reset";
  smsStatus = "Chưa gửi";
  callStatus = "Chưa gọi";
  setAlarmHardware(false);

  addDetectionLog("Hệ thống đã được reset");

  showOLED("SAFE / OK", "State: NORMAL", "IP: " + systemIP);

  server.send(200, "text/plain; charset=utf-8", "Đã reset");
}

// =======================
// ROUTE CLEAR WIFI
// =======================
void handleClearWiFi() {
  clearWiFi();
  server.send(200, "text/plain; charset=utf-8", "Đã xóa WiFi. ESP32 sẽ khởi động lại.");
  delay(2000);
  ESP.restart();
}

// =======================
// BAT AP CAI DAT WIFI
// =======================
void startSetupAP() {
  setupMode = true;

  WiFi.mode(WIFI_AP);
  WiFi.softAP(ap_ssid, ap_password);

  wifiStatus = "SETUP";
  systemIP = WiFi.softAPIP().toString();

  Serial.println();
  Serial.println("=== CHE DO CAI DAT WIFI ===");
  Serial.print("WiFi: ");
  Serial.println(ap_ssid);
  Serial.print("Password: ");
  Serial.println(ap_password);
  Serial.print("IP: ");
  Serial.println(systemIP);

  showOLED("WIFI SETUP", "WiFi: FALL_SETUP", "IP: 192.168.4.1");

  server.on("/", handleRoot);
  server.on("/savewifi", HTTP_POST, handleSaveWiFi);
  server.on("/data", handleData);

  server.begin();
}

// =======================
// KET NOI WIFI DA LUU
// =======================
bool connectWiFi() {
  if (savedSSID.length() == 0) {
    return false;
  }

  WiFi.mode(WIFI_STA);
  WiFi.begin(savedSSID.c_str(), savedPASS.c_str());

  Serial.println();
  Serial.print("Dang ket noi WiFi: ");
  Serial.println(savedSSID);

  showOLED("CONNECTING", savedSSID, "Please wait...");

  unsigned long startTime = millis();

  while (WiFi.status() != WL_CONNECTED && millis() - startTime < 15000) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    wifiStatus = "OK";
    systemIP = WiFi.localIP().toString();

    Serial.println("Da ket noi WiFi");
    Serial.print("IP he thong: ");
    Serial.println(systemIP);

    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer, "time.google.com", "time.nist.gov");

    Serial.println("Dang dong bo thoi gian NTP...");
    struct tm timeinfo;

    if (getLocalTime(&timeinfo)) {
      Serial.println("Da dong bo thoi gian");
    } else {
      Serial.println("Chua dong bo duoc thoi gian");
    }

    showOLED("SAFE / OK", "WiFi: OK", "IP: " + systemIP);

    return true;
  }

  Serial.println("Khong ket noi duoc WiFi");
  return false;
}

// =======================
// BAT WEB HE THONG
// =======================
void startSystemWeb() {
  setupMode = false;

  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.on("/set", handleSetState);
  server.on("/sms", handleSMS);
  server.on("/call", handleCall);
  server.on("/reset", handleReset);
  server.on("/clearwifi", handleClearWiFi);

  server.begin();

  addDetectionLog("Hệ thống đã khởi động và kết nối WiFi");

  Serial.println("WebServer he thong da san sang");
}

// =======================
// SETUP
// =======================
void setup() {
  Serial.begin(115200);
  delay(1000);

  // Khoi tao man hinh 1.8" TFT ST7735 (128x160) TRUOC
  tft.initR(INITR_BLACKTAB);
  tft.setRotation(1); // Xoay ngang 160x128 de hien thi dong chu va IP dai ro net
  tft.fillScreen(ST77XX_BLACK);

  // Khoi tao Coi (Buzzer) va LED do SAU tft.initR de khong bi SPI ghi de
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_RED_PIN, OUTPUT);
  setAlarmHardware(false);

  showOLED("SM-LAB", "TFT 1.8 OK", "Starting...");

  loadWiFi();

  if (connectWiFi()) {
    startSystemWeb();
  } else {
    startSetupAP();
  }
}

// =======================
// LOOP
// =======================
void loop() {
  server.handleClient();

  if (!setupMode) {
    if (WiFi.status() == WL_CONNECTED) {
      wifiStatus = "OK";
    } else {
      wifiStatus = "MAT KET NOI";
    }
  }
}