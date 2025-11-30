import React from 'react';
import ProfileSettings from '../components/user-preferences/ProfileSettings';
import Customization from '../components/user-preferences/Customization';

const UserPreferences: React.FC = () => {
  return (
    <div>
      <h1>User Preferences</h1>
      <ProfileSettings />
      <Customization />
    </div>
  );
};

export default UserPreferences;
