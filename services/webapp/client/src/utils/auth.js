export const getToken = () => {
  return localStorage.getItem('token');
};

export const getAuthHeaders = () => {
  const token = getToken();
  if (token) {
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    };
  }
  return {
    'Content-Type': 'application/json',
  };
};
