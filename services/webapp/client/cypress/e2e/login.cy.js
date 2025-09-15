describe('Login', () => {
  it('should log in successfully', () => {
    cy.visit('http://localhost:80');
    cy.get('input[type="text"]').type('testuser');
    cy.get('input[type="password"]').type('password');
    cy.get('button[type="submit"]').click();
    cy.contains('Dashboard').should('be.visible');
  });
});
