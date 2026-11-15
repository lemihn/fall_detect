# Dashboard agent (chạy trên Raspberry Pi)

Chương trình nhỏ chạy **song song** với `detect_v3/4_realtime.py` để đưa dữ liệu lên website quản trị. Agent **không sửa code nhận diện**: nó chỉ đọc các link sẵn có của Flask (`/status`, `/video_feed`) và của ESP32 (`/data`).

```
4_realtime.py (Flask :5000) ──localhost──► dashboard_agent ──► Supabase (heartbeat, ca ngã, ảnh)
ESP32 /data ──────────────────LAN────────►       │
                                                 └─ gateway :8080 ◄── cloudflared ◄── Website (camera live)
```

## Agent làm gì

| Việc | Chi tiết |
|---|---|
| Heartbeat | Mỗi 10 giây (và ngay khi trạng thái đổi), cập nhật bảng `devices`: trạng thái, FPS, số người, link camera, tình trạng hộp ESP32. |
| Phát hiện ca ngã | Khi `persons[i].telegram_sent` của `/status` chuyển sang `true` (cùng lúc với Telegram), ghi 1 dòng vào `events` kèm ảnh chụp lên bucket `snapshots`. |
| Mất mạng | Ca ngã được lưu vào `data/outbox.sqlite3` và gửi bù khi có mạng lại. Gửi lại nhiều lần cũng không bị trùng. |
| Camera live | Gateway `:8080` chỉ cho xem khi website gửi kèm token đăng nhập của người có tên trong `dashboard_users`. Agent lấy video từ Flask đúng 1 lần rồi chia cho mọi người xem, giới hạn FPS và số người xem. |
| Đưa ra Internet | Tự chạy Cloudflare quick tunnel và ghi URL vào `devices.stream_url`, website đọc URL từ đó. |

Agent chỉ cần thư viện `requests`, thư viện này `4_realtime.py` cũng đang dùng.

## Endpoint của gateway

| Endpoint | Mô tả |
|---|---|
| `GET /health` | Kiểm tra gateway còn sống (không cần token) |
| `GET /stream?token=<access_token>` | Video MJPEG, dùng trực tiếp trong `<img src>` |
| `GET /snapshot?token=<access_token>` | 1 ảnh JPEG |
| `GET /api/status` | `/status` của Pi (header `Authorization: Bearer <access_token>`) |
| `POST /api/reset_alert` | Tắt báo động |
| `POST /api/switch_camera` | Body `{"camera": 1}` |

## Cài đặt trên Pi

**Yêu cầu:** project Supabase đã chạy `supabase/schema.sql`, và `4_realtime.py` đang chạy với Flask ở cổng 5000.

### 1. Cấu hình

```bash
cd ~/fall_detect
cp dashboard_agent/.env.example dashboard_agent/.env
nano dashboard_agent/.env   # điền SUPABASE_URL, SUPABASE_SERVICE_KEY, DEVICE_ID, ESP32_IP...
```

### 2. Cài cloudflared (để xem camera từ xa)

Kiểm tra kiến trúc bằng lệnh `uname -m`: ra `aarch64` là 64-bit, ra `armv7l` là 32-bit.

```bash
# 64-bit
curl -L -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64
# 32-bit: thay bằng cloudflared-linux-arm
sudo install -m 755 cloudflared /usr/local/bin/cloudflared
cloudflared --version
```

Quick tunnel không chạy được nếu có file `~/.cloudflared/config.yml`. Nếu có file đó thì đổi tên nó đi.

### 3. Chạy thử

```bash
source .venv/bin/activate          # môi trường Python đang dùng cho 4_realtime.py
pip install -r dashboard_agent/requirements.txt
python dashboard_agent/agent.py
```

Log đúng sẽ có các dòng: `Da ket noi /status cua Pi`, `Thiet bi 'pi-01' da co tren Supabase`, `Link camera: https://....trycloudflare.com`.

### 4. Tự chạy khi Pi khởi động

Sửa `User`, `WorkingDirectory` và `ExecStart` trong `systemd/dashboard-agent.service` cho đúng với Pi, rồi chạy:

```bash
sudo cp dashboard_agent/systemd/dashboard-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dashboard-agent
journalctl -u dashboard-agent -f    # xem log
```

## Chạy thử trên laptop (không cần Pi)

```bash
# Terminal 1: chạy nhận diện với video có sẵn
python detect_v3/4_realtime.py --source duong_dan_video_te_nga.mp4
# Terminal 2: chạy agent (.env đặt TUNNEL=off nếu laptop chưa cài cloudflared)
python dashboard_agent/agent.py
```

Chạy test logic (không cần mạng):

```bash
python -m unittest discover -s dashboard_agent/tests -v
```

## Lưu ý

- `SUPABASE_SERVICE_KEY` có toàn quyền với database. Chỉ để trong `dashboard_agent/.env` trên Pi (file này đã được `.gitignore` bỏ qua).
- Ảnh hiện trường lấy từ `/video_feed` nên **có vẽ khung xương**. Telegram gửi ảnh gốc, vì ảnh gốc chỉ có bên trong `4_realtime.py`.
- Agent phụ thuộc vào các trường trong `/status`: `state`, `alarm`, `fps`, `camera`, `persons[].id`, `persons[].since`, `persons[].fall_time`, `persons[].telegram_sent`. Nếu đổi tên các trường này thì cần báo người làm dashboard.
- Mỗi lần cloudflared khởi động lại thì URL quick tunnel sẽ đổi. Agent tự cập nhật URL mới lên Supabase.
