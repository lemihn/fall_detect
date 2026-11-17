import { Icon } from './Icon'

/** Logo + ten SMLab, dung chung cho sidebar va trang dang nhap */
export function Brand({ sub = 'Giám sát té ngã' }: { sub?: string }) {
  return (
    <div className="brand">
      <div className="brand-mark"><Icon name="logo" strokeWidth={2.2} /></div>
      <div>
        <div className="brand-name">SM<span>Lab</span></div>
        <div className="brand-sub">{sub}</div>
      </div>
    </div>
  )
}
