import React, { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { getAuthHeaders } from './utils/auth';

// --- Components ---

const PortfolioList = ({ portfolios, onSelectPortfolio, onEdit, onDelete }) => {
  const { t } = useTranslation();
  return (
    <div className="Card">
      <h3>{t('portfolioManagement.myPortfolios')}</h3>
      <ul>
        {portfolios.map(p => (
          <li key={p.id}>
            <span onClick={() => onSelectPortfolio(p)} style={{ cursor: 'pointer', textDecoration: 'underline' }}>
              {p.name}
            </span>
            <div>
              <button onClick={() => onEdit(p)}>{t('portfolioManagement.edit')}</button>
              <button onClick={() => onDelete(p.id)}>{t('portfolioManagement.delete')}</button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

const AssetForm = ({ asset, onSave, onCancel }) => {
  const { t } = useTranslation();
  const [symbol, setSymbol] = useState(asset ? asset.symbol : '');
  const [quantity, setQuantity] = useState(asset ? asset.quantity : '');
  const [purchasePrice, setPurchasePrice] = useState(asset ? asset.purchase_price : '');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSave({ ...asset, symbol, quantity: parseFloat(quantity), purchase_price: parseFloat(purchasePrice) });
  };

  return (
    <form onSubmit={handleSubmit}>
      <input type="text" value={symbol} onChange={e => setSymbol(e.target.value)} placeholder={t('portfolioManagement.symbol')} required />
      <input type="number" value={quantity} onChange={e => setQuantity(e.target.value)} placeholder={t('portfolioManagement.quantity')} required />
      <input type="number" value={purchasePrice} onChange={e => setPurchasePrice(e.target.value)} placeholder={t('portfolioManagement.purchasePrice')} required />
      <button type="submit">{t('portfolioManagement.save')}</button>
      <button type="button" onClick={onCancel}>{t('portfolioManagement.cancel')}</button>
    </form>
  );
};

const PortfolioDetail = ({ portfolio, onClear, onRefresh }) => {
  const { t } = useTranslation();
  const [editingAsset, setEditingAsset] = useState(null);

  const handleSaveAsset = async (asset) => {
    const headers = await getAuthHeaders();
    const url = asset.id
      ? `/api/portfolios/${portfolio.id}/assets/${asset.id}`
      : `/api/portfolios/${portfolio.id}/assets`;
    const method = asset.id ? 'PUT' : 'POST';

    await fetch(url, {
      method,
      headers,
      body: JSON.stringify(asset),
    });
    setEditingAsset(null);
    onRefresh();
  };

  const handleDeleteAsset = async (assetId) => {
    if (window.confirm(t('portfolioManagement.deleteAssetConfirm'))) {
        const headers = await getAuthHeaders();
        await fetch(`/api/portfolios/${portfolio.id}/assets/${assetId}`, {
            method: 'DELETE',
            headers,
        });
        onRefresh();
    }
  };

  return (
    <div className="Card">
      <button onClick={onClear}>{t('portfolioManagement.backToPortfolios')}</button>
      <h3>{portfolio.name}</h3>
      <h4>{t('portfolioManagement.assets')}</h4>
      <table>
        <thead>
          <tr>
            <th>{t('portfolioManagement.symbol')}</th>
            <th>{t('portfolioManagement.quantity')}</th>
            <th>{t('portfolioManagement.purchasePrice')}</th>
            <th>{t('portfolioManagement.actions')}</th>
          </tr>
        </thead>
        <tbody>
          {portfolio.assets && portfolio.assets.map(asset => (
            <tr key={asset.id}>
              <td>{asset.symbol}</td>
              <td>{asset.quantity}</td>
              <td>${asset.purchase_price.toFixed(2)}</td>
              <td>
                <button onClick={() => setEditingAsset(asset)}>{t('portfolioManagement.edit')}</button>
                <button onClick={() => handleDeleteAsset(asset.id)}>{t('portfolioManagement.delete')}</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>{editingAsset && editingAsset.id ? t('portfolioManagement.editAsset') : t('portfolioManagement.addAsset')}</h4>
      {editingAsset ? (
        <AssetForm asset={editingAsset} onSave={handleSaveAsset} onCancel={() => setEditingAsset(null)} />
      ) : (
        <button onClick={() => setEditingAsset({})}>{t('portfolioManagement.addAsset')}</button>
      )}
    </div>
  );
};

const PortfolioForm = ({ portfolio, onSave, onCancel }) => {
    const { t } = useTranslation();
    const [name, setName] = useState(portfolio ? portfolio.name : '');

    const handleSubmit = (e) => {
        e.preventDefault();
        onSave({ ...portfolio, name });
    };

    return (
        <div className="Card">
            <h3>{portfolio ? t('portfolioManagement.edit') : t('portfolioManagement.createPortfolio')}</h3>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder={t('portfolioManagement.title')}
                    required
                />
                <button type="submit">{t('portfolioManagement.save')}</button>
                {onCancel && <button type="button" onClick={onCancel}>{t('portfolioManagement.cancel')}</button>}
            </form>
        </div>
    )
}


// --- Main Component ---

function PortfolioManagement() {
  const { t } = useTranslation();
  const [portfolios, setPortfolios] = useState([]);
  const [selectedPortfolio, setSelectedPortfolio] = useState(null);
  const [editingPortfolio, setEditingPortfolio] = useState(null);

  const fetchPortfolios = useCallback(async () => {
    const headers = await getAuthHeaders();
    const response = await fetch('/api/portfolios', { headers });
    const data = await response.json();
    setPortfolios(data);
  }, []);

  const fetchPortfolioDetails = useCallback(async (portfolio) => {
    const headers = await getAuthHeaders();
    const response = await fetch(`/api/portfolios/${portfolio.id}`, { headers });
    const data = await response.json();
    setSelectedPortfolio(data);
  }, []);


  useEffect(() => {
    fetchPortfolios();
  }, [fetchPortfolios]);

  const handleSelectPortfolio = (portfolio) => {
    fetchPortfolioDetails(portfolio);
  };

  const handleClearSelection = () => {
    setSelectedPortfolio(null);
  };

  const handleSavePortfolio = async (portfolio) => {
    const headers = await getAuthHeaders();
    const url = portfolio.id ? `/api/portfolios/${portfolio.id}` : '/api/portfolios';
    const method = portfolio.id ? 'PUT' : 'POST';

    await fetch(url, {
        method,
        headers,
        body: JSON.stringify({ name: portfolio.name }),
    });

    setEditingPortfolio(null);
    fetchPortfolios();
  }

  const handleDeletePortfolio = async (id) => {
    if (window.confirm(t('portfolioManagement.deletePortfolioConfirm'))) {
        const headers = await getAuthHeaders();
        await fetch(`/api/portfolios/${id}`, {
            method: 'DELETE',
            headers,
        });
        fetchPortfolios();
    }
  };

  if (selectedPortfolio) {
    return <PortfolioDetail portfolio={selectedPortfolio} onClear={handleClearSelection} onRefresh={() => fetchPortfolioDetails(selectedPortfolio)} />;
  }

  return (
    <div>
      <h2>{t('portfolioManagement.title')}</h2>
      <PortfolioList
        portfolios={portfolios}
        onSelectPortfolio={handleSelectPortfolio}
        onEdit={setEditingPortfolio}
        onDelete={handleDeletePortfolio}
      />
      {editingPortfolio ? (
          <PortfolioForm portfolio={editingPortfolio} onSave={handleSavePortfolio} onCancel={() => setEditingPortfolio(null)} />
      ) : (
          <button onClick={() => setEditingPortfolio({})}>{t('portfolioManagement.createPortfolio')}</button>
      )}
    </div>
  );
}

export default PortfolioManagement;
