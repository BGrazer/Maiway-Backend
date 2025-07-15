def calculate_fare(mode: str, distance_km: float, passenger_type: str = 'regular') -> float:
    """
    Calculate fare based on mode and distance
    Args:
        mode: Transport mode (LRT, Bus, Jeep, Walking)
        distance_km: Distance in kilometers
        passenger_type: Type of passenger (regular, student, senior, etc.)
    Returns:
        Fare amount in pesos
    """
    try:
        # Walking is always free
        if mode == 'Walking':
            return 0.0
        # Discount factor for non-regular passengers
        discount = 1.0
        if passenger_type and passenger_type.lower() != 'regular':
            discount = 0.8  # 20% off
        # For other modes, use distance-based fare calculation
        if mode == 'LRT':
            # LRT has fixed fare structure
            if distance_km <= 4.0:
                return 20.0 * discount
            elif distance_km <= 8.0:
                return 25.0 * discount
            else:
                return 30.0 * discount
        elif mode == 'Bus':
            # Bus has distance-based fare
            if distance_km <= 5.0:
                return 15.0 * discount
            elif distance_km <= 10.0:
                return 18.0 * discount
            else:
                return 20.0 * discount
        elif mode == 'Jeep':
            # Jeep has fixed fare
            return 15.0 * discount
        else:
            # Default fare for unknown modes
            return 10.0
    except Exception:
        # Return default fare
        if mode == 'LRT':
            return 25.0
        elif mode == 'Bus':
            return 15.0
        elif mode == 'Jeep':
            return 15.0
        else:
            return 10.0 