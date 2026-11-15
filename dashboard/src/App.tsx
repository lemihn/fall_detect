import { Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider, useAuth } from './lib/auth'
import { DataProvider } from './lib/data'
import { configError } from './lib/supabase'
import { Layout } from './components/Layout'
import { ToastProvider } from './components/ui'
import { LoginPage } from './pages/Login'
import { OverviewPage } from './pages/Overview'
import { CameraPage } from './pages/Camera'
import { AlertsPage } from './pages/Alerts'
import { HistoryPage } from './pages/History'
import { DevicesPage } from './pages/Devices'
import { SettingsPage } from './pages/Settings'

function Gate() {
  const { loading, session, profile, signOut } = useAuth()
  if (loading) return <div className="center-page"><div><div className="spinner" />Đang tải…</div></div>
  if (!session) return <LoginPage />
  if (!profile) {
    return (
      <div className="center-page">
        <div className="card">
          <h1>Tài khoản chưa được cấp quyền</h1>
          <p className="muted" style={{ marginBottom: 16 }}>
            Tài khoản <b>{session.user.email}</b> đã đăng nhập nhưng chưa có trong danh sách người dùng dashboard. Liên hệ quản trị viên để được cấp quyền.
          </p>
          <button className="btn btn-outline" onClick={signOut}>Đăng xuất</button>
        </div>
      </div>
    )
  }
  return (
    <DataProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<OverviewPage />} />
          <Route path="camera" element={<CameraPage />} />
          <Route path="alerts" element={<AlertsPage />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="devices" element={<DevicesPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </DataProvider>
  )
}

export function App() {
  if (configError) {
    return <div className="center-page"><div className="card"><h1>Thiếu cấu hình</h1><p>{configError}</p></div></div>
  }
  return (
    <ToastProvider>
      <AuthProvider>
        <Gate />
      </AuthProvider>
    </ToastProvider>
  )
}
