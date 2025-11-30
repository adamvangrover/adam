describe('Simulations', () => {
  beforeEach(() => {
    cy.visit('http://localhost:80');
    cy.get('input[type="text"]').type('testuser');
    cy.get('input[type="password"]').type('password');
    cy.get('button[type="submit"]').click();
    cy.contains('Simulations').click();
  });

  it('should run a simulation', () => {
    cy.get('button').contains('Run Simulation').first().click();
    cy.contains('Task Status:').should('be.visible');
  });
});
