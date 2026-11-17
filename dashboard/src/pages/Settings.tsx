import { useState } from 'react'
import { Icon } from '../components/Icon'
import { useToast } from '../components/ui'
import { displayName, useAuth } from '../lib/auth'
import { playAlarm, prefs, requestNotifyPermission } from '../lib/notify'

export function SettingsPage() {
  const { profile, session, signOut } = useAuth()
  const toast = useToast()
  const [sound, setSound] = useState(prefs.sound)
  const [notify, setNotify] = useState(prefs.notify && 'Notification' in window && Notification.permission === 'granted')

  async function toggleNotify() {
    if (notify) { prefs.notify = false; setNotify(false); return }
    const ok = await requestNotifyPermission()
    prefs.notify = ok
    setNotify(ok)
    toast(ok ? 'Đã bật thông báo trình duyệt' : 'Trình duyệt chưa cho phép thông báo. Bật lại trong cài đặt trang web.', !ok)
  }

  return (
    <div className="grid-2" style={{ alignItems: 'start' }}>
      <div className="card">
        <div className="card-head"><h2>Cảnh báo trên thiết bị này</h2></div>
        <div className="switch-row">
          <div><b>Chuông khi có ca ngã mới</b><span>Phát âm thanh cảnh báo khi trang đang mở</span></div>
          <button className="btn btn-sm btn-outline" onClick={playAlarm}>Nghe thử</button>
          <button className={`toggle${sound ? ' on' : ''}`} aria-label="Bật/tắt chuông" onClick={() => { prefs.sound = !sound; setSound(!sound) }} />
        </div>
        <div className="switch-row">
          <div><b>Thông báo trình duyệt</b><span>Hiện thông báo kể cả khi đang ở tab khác</span></div>
          <button className={`toggle${notify ? ' on' : ''}`} aria-label="Bật/tắt thông báo" onClick={toggleNotify} />
        </div>
      </div>
      <div className="card">
        <div className="card-head"><h2>Tài khoản</h2></div>
        <div className="card-body">
          <dl className="kv">
            <dt>Tên</dt><dd>{displayName(profile, session?.user.email)}</dd>
            <dt>Email</dt><dd>{session?.user.email}</dd>
            <dt>Vai trò</dt><dd>{profile?.role === 'admin' ? 'Quản trị viên' : 'Người theo dõi'}</dd>
          </dl>
          <button className="btn btn-outline" style={{ marginTop: 16 }} onClick={signOut}><Icon name="logout" />Đăng xuất</button>
        </div>
      </div>
    </div>
  )
}
