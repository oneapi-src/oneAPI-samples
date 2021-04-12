#ifndef __CONSTANTS_HPP__
#define __CONSTANTS_HPP__

// Allow certain design parameters to be defined on the command line but
// default to certain values
#ifndef ROWS_COMPONENT
#define ROWS_COMPONENT 40
#endif

#ifndef COLS_COMPONENT
#define COLS_COMPONENT 64 /*8*/
#endif

#ifndef NUM_STEER
#define NUM_STEER 25
#endif

// constants
constexpr int kRowsComponent = ROWS_COMPONENT;
constexpr int kColsComponent = COLS_COMPONENT;
constexpr int kNumSteer = NUM_STEER;
constexpr int kNumInput = kRowsComponent;
constexpr int kRComponent = kColsComponent;

constexpr int kASize = kRowsComponent * kColsComponent;
constexpr int kXSize = kNumSteer * kNumInput * kColsComponent;

constexpr int kRandSeed = 1138;
constexpr int kRandSeedMax = 20;
constexpr int kRandSeedMin = -10;

#endif /* __CONSTANTS_HPP__ */
