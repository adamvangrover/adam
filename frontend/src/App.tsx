import React from 'react';
import { Routes, Route, NavLink } from 'react-router-dom';
import CompaniesPage from './pages/CompaniesPage';
import CompanyDetailPage from './pages/CompanyDetailPage';
import HomePage from './pages/HomePage'; // A simple home page

function App() {
  return (
    <>
      <header>
        <h1>Narrative Library Explorer</h1>
        <nav>
          <ul>
            <li><NavLink to="/" end className={({isActive}) => isActive ? "active" : ""}>Home</NavLink></li>
            <li><NavLink to="/companies" className={({isActive}) => isActive ? "active" : ""}>Companies</NavLink></li>
          </ul>
        </nav>
      </header>
      <main className="container">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/companies" element={<CompaniesPage />} />
          <Route path="/companies/:companyId" element={<CompanyDetailPage />} />
        </Routes>
      </main>
      <footer>
        <p style={{ textAlign: 'center', marginTop: '30px', color: '#8a8d91', fontSize: '14px' }}>
          Narrative Library &copy; {new Date().getFullYear()}
        </p>
      </footer>
    </>
  );
}

export default App;
