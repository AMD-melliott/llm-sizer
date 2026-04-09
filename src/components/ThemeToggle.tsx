import { Sun, Moon, Monitor } from 'lucide-react';
import useThemeStore from '../store/useThemeStore';

export default function ThemeToggle() {
  const { theme, setTheme } = useThemeStore();

  const options = [
    { value: 'light' as const, icon: Sun, label: 'Light' },
    { value: 'dark' as const, icon: Moon, label: 'Dark' },
    { value: 'system' as const, icon: Monitor, label: 'System' },
  ];

  return (
    <div className="flex items-center rounded-lg p-0.5" style={{ backgroundColor: 'var(--surface-inset)', border: '1px solid var(--border)' }}>
      {options.map(({ value, icon: Icon, label }) => (
        <button
          key={value}
          onClick={() => setTheme(value)}
          title={label}
          className="relative p-1.5 rounded-md transition-all duration-200"
          style={{
            backgroundColor: theme === value ? 'var(--surface)' : 'transparent',
            color: theme === value ? 'var(--accent)' : 'var(--text-muted)',
            boxShadow: theme === value ? '0 1px 3px var(--shadow-color)' : 'none',
          }}
        >
          <Icon className="w-4 h-4" />
        </button>
      ))}
    </div>
  );
}
