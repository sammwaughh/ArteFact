// viewer/src/App.tsx
/**
 * Root component for the v2 SPA.
 * For now we mount the single-page upload / inference flow.
 * (Routing can be added later with React-Router if we grow beyond one page.)
 */
import HomePage from './pages/HomePage';
import AppHeader from './components/AppHeader';

 export default function App() {
   return (
     <>
       <AppHeader />
       <HomePage />
     </>
   );
 }
