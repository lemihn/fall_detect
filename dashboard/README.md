# Website quản trị (dashboard)

Website theo dõi hệ thống phát hiện té ngã: cảnh báo realtime, camera trực tiếp, lịch sử và thống kê. Dữ liệu lấy từ Supabase (xem `supabase/`). Video live lấy từ `dashboard_agent` chạy trên Pi.

**Công nghệ:** React 19 + Vite + TypeScript + supabase-js. Biểu đồ vẽ bằng SVG, không dùng thư viện biểu đồ.

## Các trang

| Trang | Chức năng |
|---|---|
| Đăng nhập | Supabase Auth. Chỉ tài khoản có trong `dashboard_users` mới vào được |
| Tổng quan | 4 ô KPI, camera của thiết bị đang báo ngã, hoạt động gần đây, bảng thiết bị, biểu đồ 7/30 ngày |
| Camera trực tiếp | Video live qua gateway, xác suất 4 tư thế cập nhật mỗi giây, tắt báo động, đổi camera, ảnh xem trước của thiết bị khác |
| Cảnh báo | Danh sách ca ngã (lọc, tìm kiếm). Chi tiết có ảnh, dòng thời gian, ghi chú, xác nhận / đã xử lý / mở lại |
| Lịch sử & thống kê | 7/30/90 ngày: KPI, số ca theo ngày, theo thiết bị, khung giờ hay xảy ra, xuất CSV |
| Thiết bị | Tình trạng từng Pi. Admin sửa được tên và vị trí |
| Cài đặt | Bật/tắt chuông, thông báo trình duyệt, đăng xuất |

Khi có ca ngã mới, **dải đỏ** hiện ở đầu mọi trang, kèm chuông và thông báo trình duyệt (nếu đã bật), không cần tải lại trang. Màn hình dưới 900px dùng giao diện mobile, có thanh menu dưới đáy.

## Chạy trên máy

```bash
cd dashboard
cp .env.example .env      # điền VITE_SUPABASE_URL và VITE_SUPABASE_ANON_KEY
npm install
npm run dev               # http://localhost:5173
```

`VITE_SUPABASE_ANON_KEY` là **publishable / anon key**, được thiết kế để công khai trên web. Quyền truy cập được bảo vệ bằng RLS trong `supabase/schema.sql`. **Không** dùng secret / service_role key ở đây.

## Deploy lên Vercel

1. Vercel → **Add New → Project** → chọn repo `fall_detect`, branch `quangtien`.
2. **Root Directory:** `dashboard`. Framework tự nhận là Vite.
3. **Environment Variables:** thêm `VITE_SUPABASE_URL` và `VITE_SUPABASE_ANON_KEY`.
4. Bấm Deploy. File `vercel.json` đã cấu hình để các đường dẫn như `/alerts` không bị 404.

Website chạy bằng `https`, nên camera live cần link `https` từ Cloudflare Tunnel. Link `http://IP-nội-bộ` sẽ bị trình duyệt chặn.

## Cấu trúc

```
src/
  lib/        supabase, auth, dữ liệu realtime (data.tsx), gateway camera, định dạng, thống kê, thông báo
  components/ Layout (sidebar, topbar, menu mobile, dải cảnh báo), ui, charts, Icon
  pages/      Login, Overview, Camera, Alerts, History, Devices, Settings
  styles/     base.css (theo mẫu giao diện đã duyệt), app.css (tương tác, mobile)
```
