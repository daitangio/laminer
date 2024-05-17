def build_file_prefix():
    """
    Build a simple date-based-prefix based also on current hour
    """
    today = datetime.today().date()
    formatted_date = today.strftime('%Y-%m-%d-%H')
    return formatted_date