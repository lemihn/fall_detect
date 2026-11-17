# Supabase cho Dashboard phát hiện té ngã

Thư mục này chứa database cho website quản trị (phần dashboard). Website và `dashboard_agent` trên Pi đều dùng chung database này. Thư mục này không ảnh hưởng tới phần nhận diện hay firmware ESP32.

## Cấu trúc

| Thành phần | Mục đích | Ai ghi | Ai đọc |
|---|---|---|---|
| `dashboard_users` | Danh sách tài khoản được dùng dashboard (`admin` / `viewer`) | Admin (qua SQL) | Chính người đó, admin |
| `devices` | Mỗi Pi là 1 dòng: trạng thái hiện tại, heartbeat, link camera, hộp ESP32 | Agent trên Pi | Website |
| `events` | Mỗi ca té ngã là 1 dòng: thời điểm, thời gian bất động, ảnh, trạng thái xử lý | Agent tạo, website cập nhật trạng thái | Website |
| Bucket `snapshots` | Ảnh hiện trường (private, xem qua signed URL) | Agent | Website |

Trạng thái xử lý một ca ngã đi theo thứ tự `new` → `acknowledged` (đã xác nhận) → `resolved` (đã xử lý). Database tự ghi lại ai đổi trạng thái và lúc nào.

## Phân quyền

- **Chưa đăng nhập (anon key):** không đọc được gì.
- **Đã đăng nhập nhưng không có tên trong `dashboard_users`:** không đọc được gì, kể cả khi ai đó tự đăng ký được tài khoản.
- **`viewer`:** xem tất cả, đổi trạng thái và ghi chú của ca ngã.
- **`admin`:** như viewer, thêm quyền đổi tên và vị trí thiết bị.
- **Agent trên Pi (`service_role` key):** ghi được mọi thứ. **Key này chỉ để trong `.env` trên Pi, tuyệt đối không đưa lên git hay website.**

## Cài đặt

1. Tạo project Supabase (region Asia-Pacific / Singapore).
2. Mở **SQL Editor**, dán toàn bộ nội dung `schema.sql` vào rồi bấm **Run**. File này chạy lại nhiều lần cũng được.
3. Tạo tài khoản đăng nhập: **Authentication → Users → Add user → Create new user**, nhập email và mật khẩu, tick **Auto Confirm User**.
4. Cấp quyền dashboard cho tài khoản đó trong SQL Editor (thay email):

   ```sql
   insert into public.dashboard_users (user_id, email, role)
   select id, email, 'admin' from auth.users where email = 'ban@example.com'
   on conflict (user_id) do update set role = excluded.role;
   ```

5. (Nên làm) **Authentication → Sign In / Providers → tắt "Allow new users to sign up"**, để chỉ admin tạo được tài khoản.

## Lấy key cho các phần khác

Vào **Project Settings → API**:

- `Project URL` và `anon` / publishable key → dùng cho website (`dashboard/.env`).
- `service_role` / secret key → **chỉ** dùng cho agent trên Pi (`dashboard_agent/.env`).
