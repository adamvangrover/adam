describe('Portfolio Management', () => {
  beforeEach(() => {
    cy.visit('http://localhost:80');
    cy.get('input[type="text"]').type('testuser');
    cy.get('input[type="password"]').type('password');
    cy.get('button[type="submit"]').click();
    cy.contains('Portfolio Management').click();
  });

  it('should create a new portfolio', () => {
    const portfolioName = 'My New Portfolio';
    cy.get('input[placeholder="Portfolio Name"]').type(portfolioName);
    cy.get('button[type="submit"]').click();
    cy.contains(portfolioName).should('be.visible');
  });
});
