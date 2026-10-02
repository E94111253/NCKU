
`timescale 1ns/1ps

module TTT_tb;

    reg clk;
    reg reset;
    reg Up;
    reg Down;
    reg Left;
    reg Right;
    reg sel;

    wire [5:0] row;
    wire [11:0] col;
    wire [1:0] Win;

    TTT uut (
        .clk(clk),
        .reset(reset),
        .Up(Up),
        .Down(Down),
        .Left(Left),
        .Right(Right),
        .sel(sel),
        .row(row),
        .col(col),
        .Win(Win)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        reset = 0;
        Up = 0;
        Down = 0;
        Left = 0;
        Right = 0;
        sel = 0;

        $monitor("Time: %d, reset: %b, Up: %b, Down: %b, Left: %b, Right: %b, sel: %b, row: %b, col: %b, Win: %b", $time, reset, Up, Down, Left, Right, sel, row, col, Win);

        reset = 1;
        #10;
        reset = 0;
        
        //O:(1,1)
        #10;
        sel = 1;
        #10;
        sel = 0;
        #10;
        
        //X:(2,1)
        Right = 1;
        #10;
        Right = 0;
        sel = 1;
        #10;
        sel = 0;
        #10;
        
        //O:(0,1)
        Right = 1;
        #10;
        Right = 0;
        sel = 1;
        #10;
        sel = 0;
        #10;
        
        //X:(0,2)
        Up = 1;
        #10;
        Up = 0;
        sel = 1;
        #10;
        sel = 0;
        #10;
        
        //O:(0,0)
        Up = 1;
        #10;
        Up = 0;
        sel = 1;
        #10;
        sel = 0;
        #10;
        
        //X:(1,2)
        Down = 1;
        #10;
        Down = 0;
        #10;
        Right = 1;
        #10;
        Right = 0;
        sel = 1;
        #10;
        sel = 0;
        #10;
        
        //O:(1,1)
        Down = 1;
        #10;
        Down = 0;
        sel = 1;
        #10;
        sel = 0;
        #10;
        
        //X:(2,2) X Win
        Up = 1;
        #10;
        Up = 0;
        #10;
        Right = 1;
        #10;
        Right = 0;
        sel = 1;
        #10;
        sel = 0;
        #10;
        
        //O:(2,1) No Change
        Down = 1;
        #10;
        Down = 0;
        sel = 1;
        #10;
        sel = 0;
        #300;
        
        reset = 1;
        #10;
        reset = 0;
        #30;
        
        //O:(1,1)
        #10;
        sel = 1;
        #10;
        sel = 0;
        #10;
        
        //X:(2,1)
        Right = 1;
        #10;
        Right = 0;
        sel = 1;
        #10;
        sel = 0;
        #10;
        
        //O:(0,1)
        Right = 1;
        #10;
        Right = 0;
        sel = 1;
        #10;
        sel = 0;
        #10;
        
        //X:(0,2)
        Up = 1;
        #10;
        Up = 0;
        sel = 1;
        #10;
        sel = 0;
        #10;
        
        //O:(0,0)
        Up = 1;
        #10;
        Up = 0;
        sel = 1;
        #10;
        sel = 0;
        #10;
        
        //X:(1,2)
        Down = 1;
        #10;
        Down = 0;
        #10;
        Right = 1;
        #10;
        Right = 0;
        sel = 1;
        #10;
        sel = 0;
        #10;
        
        //O:(1,1)
        Down = 1;
        #10;
        Down = 0;
        sel = 1;
        #10;
        sel = 0;
        #10;
        
        //X:(2,2) X Win
        Up = 1;
        #10;
        Up = 0;
        #10;
        Right = 1;
        #10;
        Right = 0;
        sel = 1;
        #10;
        sel = 0;
        #10;
        
        //O:(2,1) No Change
        Down = 1;
        #10;
        Down = 0;
        sel = 1;
        #10;
        sel = 0;
        #300; 
        
        reset = 1;
        #10;
        reset = 0;
        #30;
        
        $finish;
    end

endmodule
