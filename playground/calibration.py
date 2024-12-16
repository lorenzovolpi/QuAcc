from abstention.calibration import TempScaling

if __name__ == "__main__":
    ts = TempScaling()
    bcts = TempScaling(bias_positions="all")
