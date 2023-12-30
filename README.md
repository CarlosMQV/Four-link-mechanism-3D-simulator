# Four-link-mechanism-3D-simulator

**This is a 3D simulator of a 4-link mechanism, in which you have defined the centers of rotation and three positions of point P. You find the length of the links and a very cool animation!**

This is the simulator of a four-bar mechanism in a container that emulates a robot that collects garbage. 3 points are defined (P1, P2 and P3) through which point P (end of the pick-up) will pass and the centers where the rotating links will be.

First, these input data are processed and the link lengths are generated. Then, from those lengths, the coordinates of the points are calculated based on the angle of rotation of link O2. Subsequently, matrices are created for the moving points (A, B and P) that store all the positions they can take.

Then, all coordinates are processed so that from 3D, they are converted to 2D by multiplication with a projection matrix. Then the system is rotated, since initially the coordinates were found for a system with O2 as the center and O4 on the same horizontal axis as O2 (which does not necessarily have to be the case) and finally the system is moved, since our center of coordinates will not be O2, but the bottom left of the rectangular box.

Thus, the points are connected and a projection is generated that makes the system appear three-dimensional.

On the other hand, a window was implemented in tkinter to enter the data, and it is necessary to press the calculation button after writing the corresponding values. Thus, a screen will be displayed with the lengths of the links of interest. Then, closing the windows will open the pygame screen where you can move the mechanism.

Using the right and left arrow keys on the keyboard, the system rotates around its y axis. With the up and down arrows, the mechanism is activated and point P rises or falls according to the conditions initially specified.

Also attached are some scribbled images that were made during the development of the program. The entire program was done in one day (the day I'm writing this), so it would be great if maybe you could optimize it. If you did, feel free to let me know and send me a link to your github repository to review it with some small comments to understand it (I'm not the best at understanding other people's code :p)

So good luck and have fun!

- Querevalú Vásquez, Carlos Marcelo

Suggested data:

P1x = -15.95
P1y = 0
P2x = -14.51
P2y = 22.18
P3x = 4.62
P3y = 32.12
O2x = 4.928
O2y = 8.517
O4x = 5.15
O4y = 11.7
alpha2 = -30
alpha3 = -80

Authors:	Alvarez Sanchez Arturo Estefano (https://github.com/4rturo4Lvarez)
      		Querevalú Vásquez Carlos Marcelo (https://github.com/CarlosMQV)


             ,----------------,              ,---------,
        ,-----------------------,          ,"        ,"|
      ,"                      ,"|        ,"        ,"  |
     +-----------------------+  |      ,"        ,"    |
     |  .-----------------.  |  |     +---------+      |
     |  |                 |  |  |     | -==----'|      |
     |  |  TMMS7 Sim.     |  |  |     |         |      |
     |  |  3D > 2D        |  |  |/----|`---=    |      |
     |  |  > py main.py   |  |  |   ,/|==== ooo |      ;
     |  |                 |  |  |  // |(((( [33]|    ,"
     |  `-----------------'  |," .;'| |((((     |  ,"
     +-----------------------+  ;;  | |         |,"
        /_)______________(_/  //'   | +---------+
   ___________________________/___  `,
  /  oooooooooooooooo  .o.  oooo /,   \,"-----------
 / ==ooooooooooooooo==.o.  ooo= //   ,`\--{)B     ,"
/_==__==========__==_ooo__ooo=_/'   /___________,"
`-----------------------------'
