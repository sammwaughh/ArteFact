// viewer/src/App.tsx
/**
 * Root component for the v2 SPA.
 * For now we mount the single-page upload / inference flow.
 * (Routing can be added later with React-Router if we grow beyond one page.)
 */
import UploadPage from './pages/UploadPage';

export default function App() {
  return <UploadPage />;
}
