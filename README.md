# ArtContext Viewer

A modern web application for viewing paintings with contextual academic annotations and labels. Built with React and TypeScript, this viewer displays artwork alongside scholarly interpretations, confidence ratings, and academic source citations.

## Features

- **Interactive Art Viewing**: Full-screen painting display with responsive image scaling
- **Academic Context**: Scholarly labels and interpretations with confidence scores
- **Source Attribution**: Direct links to academic papers via DOI when available
- **Responsive Design**: Clean, professional interface that works across devices
- **Modern Tech Stack**: Built with React, TypeScript, and Chakra UI

## Architecture

The application follows a component-based architecture built on modern web technologies:

### Frontend Stack

- **React 18** - Component-based UI framework with hooks and modern patterns
- **TypeScript** - Type-safe JavaScript for better development experience
- **Chakra UI** - Modular and accessible component library for consistent design
- **Vite** - Fast build tool with hot module replacement for development

### Project Structure

```
src/
├── components/           # Reusable UI components
│   ├── PaintingViewer.tsx   # Main artwork display component
│   ├── Sidebar.tsx          # Labels and metadata panel
│   └── LabelCard.tsx        # Individual label with source info
├── types/               # TypeScript type definitions
│   └── labels.d.ts         # Data structure interfaces
└── App.tsx             # Main application component
```

### Data Architecture

- **JSON-based content**: Painting metadata and labels stored in structured JSON
- **Type-safe interfaces**: Strongly typed data models for paintings, labels, and sources
- **Dynamic content loading**: Async data fetching with error handling and loading states

## Installation

### Prerequisites

- **Node.js** (version 18 or higher)
- **npm** (comes with Node.js)

### Windows Installation

1. Install Node.js from [nodejs.org](https://nodejs.org/)
2. Open Command Prompt or PowerShell
3. Clone and setup the project:

```cmd
git clone https://github.com/sammwaughh/viewer-v1.git
cd viewer-v1
npm install
```

### Linux/macOS Installation

1. Install Node.js:

   - **Ubuntu/Debian**: `sudo apt update && sudo apt install nodejs npm`
   - **macOS with Homebrew**: `brew install node`
   - **Or download from** [nodejs.org](https://nodejs.org/)

2. Clone and setup the project:

```bash
git clone https://github.com/sammwaughh/viewer-v1.git
cd viewer-v1
npm install
```

## Usage

### Development

Start the development server with hot reload:

```bash
npm run dev
```

The application will be available at `http://localhost:5173`

### Production Build

Create an optimized production build:

```bash
npm run build
```

Built files will be in the `dist/` directory.

### Preview Production Build

Preview the production build locally:

```bash
npm run preview
```

## Development Workflow

### Code Quality

The project includes comprehensive code quality tools:

- **Linting**: `npm run lint` - ESLint with Airbnb TypeScript configuration
- **Formatting**: `npm run format` - Prettier code formatting
- **Testing**: `npm test` - Vitest test runner
- **Git Hooks**: Husky pre-commit hooks ensure code quality

### Adding New Artwork

1. Add image to `public/images/`
2. Create corresponding JSON file in `public/data/` following the schema in `src/types/labels.d.ts`
3. Update the fetch URL in `App.tsx` to load your new data file

## Technology Stack

- **React 18.2** - UI framework
- **TypeScript 5.5** - Type safety
- **Vite 6.3** - Build tool and dev server
- **Chakra UI 2.8** - Component library
- **Framer Motion 10** - Animation library
- **ESLint** - Code linting with Airbnb configuration
- **Prettier** - Code formatting
- **Husky** - Git hooks
- **Vitest** - Testing framework
