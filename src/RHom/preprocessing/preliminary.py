from RHom._deps import pd
from factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

def check_stats(df: pd.DataFrame, verbosity: int = 0) -> None:
    """
    This function checks the KMO and Bartlett Sphericity of the dataframe.
    Args:
        df: The dataframe to check.
    Returns:
        None
    """
    bart = calculate_bartlett_sphericity(df)
    kmo = calculate_kmo(df)

    if verbosity > 0:
        print(interpret_bartlett(bart, df))
        print(interpret_kmo(kmo))

        return
    
   
    return {"bartlett": bart, "kmo": kmo}

def interpret_bartlett(bart, df: pd.DataFrame) -> None:
        
    interpretation_dict = {"status":"acceptable" if bart[1] < 0.05 else "unacceptable"}

    p = df.shape[1]
    degreesfreedom = p * (p - 1) / 2

    return f"χ2({int(degreesfreedom)}) = {bart[0]:.2f}, p = {bart[1]:.3f}.\nBartlett Sphericity is {interpretation_dict['status']}."

def interpret_kmo(kmo):
        
    k = kmo[1]
    
    interpretation_dict = {
        (0.9, 1): "marvelous",
        (0.8, 0.9): "meritorious",
        (0.7, 0.8): "middling",
        (0.6, 0.7): "mediocre",
        (0.5, 0.6): "miserable",
        (0, 0.5): "unacceptable"
    }

    interpretation = next((msg for (min_s, max_s), msg in interpretation_dict.items() if min_s <= k <= max_s), "KMO is invalid")

    return f"KMO = {k:.2f}, which is {interpretation} for dimension reduction."