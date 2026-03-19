import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { DashboardPage } from './DashboardPage';
import '../../index.css';

const apiBase = window.location.origin;

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <DashboardPage apiBase={apiBase} />
  </StrictMode>
);
