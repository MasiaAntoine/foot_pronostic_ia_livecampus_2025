def get_css():
    """
    Retourne le CSS pour styler l'application.
    """
    return """
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #1E3A8A;
    margin-top: 1rem;
}
.result-box {
    padding: 20px;
    border-radius: 5px;
    margin: 10px 0;
    font-weight: bold;
    text-align: center;
    font-size: 1.2rem;
}
.win {background-color: #DCFCE7; color: #166534;}
.draw {background-color: #FEF3C7; color: #92400E;}
.loss {background-color: #FEE2E2; color: #991B1B;}
</style>
"""
