describe('Analysis Tools', () => {
  beforeEach(() => {
    cy.visit('http://localhost:80');
    cy.get('input[type="text"]').type('testuser');
    cy.get('input[type="password"]').type('password');
    cy.get('button[type="submit"]').click();
    cy.contains('Analysis Tools').click();
  });

  it('should run the fundamental analyst agent', () => {
    cy.get('input[type="text"]').first().type('AAPL');
    cy.get('button').contains('Run Analysis').first().click();
    cy.contains('Output:').should('be.visible');
  });
});
