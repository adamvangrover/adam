import React, { useState, useEffect } from 'react';
import { getAuthHeaders } from './utils/auth';

function PortfolioManagement() {
  const [portfolios, setPortfolios] = useState([]);
  const [newPortfolioName, setNewPortfolioName] = useState('');
  const [editingPortfolio, setEditingPortfolio] = useState(null);

  const fetchPortfolios = () => {
    fetch('/api/portfolios', { headers: getAuthHeaders() })
      .then(res => res.json())
      .then(data => setPortfolios(data));
  };

  useEffect(() => {
    fetchPortfolios();
  }, []);

  const handleCreatePortfolio = (e) => {
    e.preventDefault();
    fetch('/api/portfolios', {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ name: newPortfolioName }),
    })
      .then(res => res.json())
      .then(() => {
        fetchPortfolios();
        setNewPortfolioName('');
      });
  };

  const handleDeletePortfolio = (id) => {
    fetch(`/api/portfolios/${id}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    }).then(() => fetchPortfolios());
  };

  const handleUpdatePortfolio = (e) => {
    e.preventDefault();
    fetch(`/api/portfolios/${editingPortfolio.id}`, {
      method: 'PUT',
      headers: getAuthHeaders(),
      body: JSON.stringify({ name: editingPortfolio.name }),
    })
      .then(res => res.json())
      .then(() => {
        fetchPortfolios();
        setEditingPortfolio(null);
      });
  };

  return (
    <div>
      <h2>Portfolio Management</h2>
      <div className="Card">
        <h3>My Portfolios</h3>
        <ul>
          {portfolios.map(p => (
            <li key={p.id}>
              {editingPortfolio && editingPortfolio.id === p.id ? (
                <form onSubmit={handleUpdatePortfolio}>
                  <input
                    type="text"
                    value={editingPortfolio.name}
                    onChange={(e) => setEditingPortfolio({ ...editingPortfolio, name: e.target.value })}
                  />
                  <button type="submit">Save</button>
                  <button onClick={() => setEditingPortfolio(null)}>Cancel</button>
                </form>
              ) : (
                <>
                  {p.name}
                  <button onClick={() => setEditingPortfolio(p)}>Edit</button>
                  <button onClick={() => handleDeletePortfolio(p.id)}>Delete</button>
                </>
              )}
            </li>
          ))}
        </ul>
      </div>
      <div className="Card">
        <h3>Create New Portfolio</h3>
        <form onSubmit={handleCreatePortfolio}>
          <input
            type="text"
            value={newPortfolioName}
            onChange={(e) => setNewPortfolioName(e.target.value)}
            placeholder="Portfolio Name"
          />
          <button type="submit">Create</button>
        </form>
      </div>
    </div>
  );
}

export default PortfolioManagement;
