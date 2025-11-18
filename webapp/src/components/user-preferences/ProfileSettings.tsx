import React from 'react';

const ProfileSettings: React.FC = () => {
  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', margin: '16px 0', borderRadius: '5px' }}>
      <h3 style={{ marginTop: 0 }}>Profile Settings</h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '10px', alignItems: 'center', maxWidth: '500px' }}>
        <label>Name:</label>
        <input type="text" defaultValue="John Doe" style={{ padding: '5px' }} />

        <label>Email:</label>
        <input type="email" defaultValue="john.doe@example.com" style={{ padding: '5px' }} />

        <label>Password:</label>
        <button>Change Password</button>

        <div style={{ gridColumn: 'span 2', textAlign: 'right', paddingTop: '10px' }}>
          <button style={{ padding: '10px 20px' }}>Save Changes</button>
        </div>
      </div>
    </div>
  );
};

export default ProfileSettings;
