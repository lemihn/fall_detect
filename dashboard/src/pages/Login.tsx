import { useState, type FormEvent } from 'react'
import { Icon } from '../components/Icon'
import { Brand } from '../components/Brand'
import { useAuth } from '../lib/auth'

export function LoginPage() {
  const { signIn } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  async function submit(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    setErr(null)
    const msg = await signIn(email.trim(), password)
    setBusy(false)
    if (msg) setErr(msg)
  }

  return (
    <div className="login">
      <section className="login-left">
        <Brand sub="Hệ thống giám sát té ngã" />
        <h2>Theo dõi an toàn người thân theo thời gian thực, mọi lúc, mọi nơi.</h2>
        <p>Nhận cảnh báo té ngã ngay khi xảy ra, xem camera trực tiếp và xác nhận xử lý từ máy tính hoặc điện thoại.</p>
        <div className="facts">
          <div><b>Thời gian thực</b><span>cảnh báo trong vài giây</span></div>
          <div><b>Camera live</b><span>xem từ xa qua kết nối bảo mật</span></div>
          <div><b>24/7</b><span>giám sát liên tục</span></div>
        </div>
        <div className="foot">© {new Date().getFullYear()} SMLab · Hệ thống phát hiện té ngã · Raspberry Pi + ESP32</div>
      </section>
      <section className="login-right">
        <form className="form" onSubmit={submit}>
          <div className="login-mobile-brand"><Brand sub="Hệ thống giám sát té ngã" /></div>
          <h1>Đăng nhập</h1>
          <div className="lead">Dùng tài khoản do quản trị viên cấp.</div>
          <div className="field">
            <label htmlFor="email">Email</label>
            <input id="email" className="input" type="email" autoComplete="username" required value={email} onChange={e => setEmail(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="pw">Mật khẩu</label>
            <input id="pw" className="input" type="password" autoComplete="current-password" required value={password} onChange={e => setPassword(e.target.value)} />
          </div>
          {err && <div className="err-box" style={{ marginBottom: 14 }}>{err}</div>}
          <button className="btn btn-primary btn-full" disabled={busy}>{busy ? 'Đang đăng nhập…' : 'Đăng nhập'}</button>
          <div className="note"><Icon name="warn" /><span>Tài khoản mới không tự đăng ký được. Liên hệ quản trị viên để được cấp quyền truy cập.</span></div>
        </form>
      </section>
    </div>
  )
}
