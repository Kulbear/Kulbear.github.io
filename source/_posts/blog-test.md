---
title: Blog Test
type: post
category: Personal
tags: [中文]
description: Testing blog plugins only. Nothing special inside.
---

## Latex Test

Inline: $\frac{y'}{1-y'} = \frac{y'}{1-y'} * \frac{m^-}{m^+}$

Single-line: $$\frac{y'}{1-y'} = \frac{y'}{1-y'} * \frac{m^-}{m^+}$$

## Code Test

```python
class Solution:
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """

        m = len(grid)  # row
        n = len(grid[0])  # col

        # m by n matrix
        dp = [[0 for _ in range(n)] for _ in range(m)]

        # these for loops work in the way that
        # filling the dp matrix row by row,
        # from left to right
        # O(m*n)
        for i in range(m):
            for j in range(n):
                # initial point, coordinate 0, 0
                if i == 0 and j == 0:
                    dp[i][j] = grid[i][j]
                # at first column, only move from the above row
                elif i > 0 and j == 0:
                    dp[i][j] = dp[i - 1][j] + grid[i][j]
                # at first row, only move from the left col
                elif i == 0 and j > 0:
                    dp[i][j] = dp[i][j - 1] + grid[i][j]
                # move from above or left
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]

        return dp[m - 1][n - 1]
```

## Chinese Test

今年下半年，中美合拍的西游记即将正式开机。我将...

## English Test

### Heal the World
```
There's a place in your heart
And I know that it is love
And this place could be much
Brighter than tomorrow
And if you really try
You'll find there's no need to cry
In this place you'll feel
There's no hurt or sorrow
There are ways to get there
If you care enough for the living
Make a little space
Make a better place
Heal the world
Make it a better place
For you and for me
And the entire human race
There are people dying
If you care enough for the living
Make it a better place
For you and for me
If you want to know why
There's love that cannot lie
Love is strong
It only cares of joyful giving
If we try we shall see
In this bliss we cannot feel
Fear of dread
We stop existing and start living
The it feels that always
Love's enough for us growing
So make a better world
Make a better place
Heal the world
Make it a better place
For you and for me
And the entire human race
There are people dying
If you care enough for the living
Make a better place for you and for me
And the dream we were conceived in
Will reveal a joyful face
And the world we once believed in
Will shine again in grace
Then why do we keep strangling life
Wound this earth, crucify its soul
Though it's plain to see
This world is heavenly
Be god's glow
We could fly so high
Let our spirits never die
In my heart I feel you are all my brothers
Create a world with no fear
Together we cry happy tears
See the nations turn their swords into plowshares
We could really get there
If you cared enough for the living
Make a little space
To make a better place
Heal the world
Make it a better place
For you and for me
And the entire human race
There are people dying 
If you care enough for the living
Make a better place for you and for me
There are people dying
If you care enough for the living
Make a better place for you and for me
You and for me
```
Songwriters: Michael Joe Jackson