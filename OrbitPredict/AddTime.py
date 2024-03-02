def AddTime(Time, addsecond):
    if addsecond != 0:
        logic = int(addsecond / abs(addsecond))  # Sign of time forward/backward
        addSecond = round(abs(addsecond))
        addMinute = addSecond // 60
        addSecond -= addMinute * 60
        addHour = addMinute // 60
        addMinute -= addHour * 60
        addDay = addHour // 24
        addHour -= addDay * 24
        newTime = Time + logic * [0, 0, addDay, addHour, addMinute, addSecond]
        newTime = AmendTime(newTime)
    else:
        newTime = Time
    return newTime

def AmendTime(startTime):
    dayOfMonth = [31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Number of days in each month (January to December)
    if startTime[1] % 4 == 0:
        dayOfMonth[2] = dayOfMonth[1] + 1

    if startTime[5] >= 60:  # Adjust seconds
        startTime[5] -= 60
        startTime[4] += 1
    elif startTime[5] < 0:
        startTime[5] = 60 + startTime[5]
        startTime[4] -= 1

    if startTime[4] >= 60:  # Adjust minutes
        startTime[4] -= 60
        startTime[3] += 1
    elif startTime[4] < 0:
        startTime[4] = 60 + startTime[4]
        startTime[3] -= 1

    if startTime[3] >= 24:  # Adjust hours
        startTime[3] -= 24
        startTime[2] += 1
    elif startTime[3] < 0:
        startTime[3] = 24 + startTime[3]
        startTime[2] -= 1

    if startTime[2] > dayOfMonth[startTime[1] + 1]:  # Adjust days
        startTime[2] -= dayOfMonth[startTime[1] + 1]
        startTime[1] += 1
    elif startTime[2] <= 0:
        startTime[2] = dayOfMonth[startTime[1]] + startTime[2]
        startTime[1] -= 1

    if startTime[1] >= 13:  # Adjust months
        startTime[1] -= 12
        startTime[0] += 1
    elif startTime[1] <= 0:
        startTime[1] = 12 + startTime[1]
        startTime[0] -= 1

    newTime = startTime
    return newTime
