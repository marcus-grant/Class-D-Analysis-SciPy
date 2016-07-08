## Class D Amplifier Analysis
- I'm designing a range of inexpensive open-sourced Class D amplifiers that rely on inexpensive microcontrollers
- I'm also interested in replacing MATLAB with Python and frameworks like SciPy, NumPy, MatPlotLib, and parallel computing libraries as I improve my abilities in Data Science
- This project combines both as I try and design the best possible Class D Amps using the least amount of cost in easy to use electronics hardware
- This folder will eventually be filled with markdown articles and their supporting codes as I update my findings
- The first will be a design using 8-bit AVR microcontrollers such as the ever popular ATMega328 found in Arduinos or the diminutive ATTiny85.
- Using microcontrollers the part count will be exceptionally low as the Triangle Wave generator, comparator, and modulation modules will all be contained within the microntroller in the form of interrupt routines servicing an ADC, a register to update the triangle wave, and timers to produce a PWM wave made from the comparison of the virtual triangle wave and the incoming audio signals.

