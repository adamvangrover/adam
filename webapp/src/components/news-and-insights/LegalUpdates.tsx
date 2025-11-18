import React from 'react';

const LegalUpdates: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Legal & Regulatory Updates</h3>
      <p>A feed of relevant legal and regulatory updates from various sources.</p>
       <ul style={{ margin: 0, paddingLeft: '20px' }}>
        <li>
          <strong>[SEC]</strong> New disclosure requirements for institutional investors.
        </li>
        <li>
          <strong>[EU]</strong> MiFID II review and potential changes.
        </li>
      </ul>
    </div>
  );
};

export default LegalUpdates;
