//cnn schematics sketch

import settings;
import graph3;

settings.render=0; // was 80
settings.antialias=16; // was 16
settings.outformat="pdf"; // was pdf
settings.prc=false;

usepackage('pifont');
usepackage("asycolors");
usepackage("amsmath");

//defaultrender.tubegranularity = 1e-3;

size(8cm);

real zoom = 4.0;

currentprojection=orthographic(
camera=(-1.5, 1.8,-1), // was camera=(-1, 2,-2.7),
//camera=(-1.5, 3,-1), // was camera=(-1, 2,-2.7),
up=(0,0,-1),
target=(0.5,0.5,0),
zoom=1);


currentlight=light(white,(2,2,2),(2,2,-2));

real plane_opacity = 0.2;
real plane_opacity2 = 0.15;

real plane_over = 0.2;
real line_over = 0.2;

real dot_size = 1.5bp;
real arrow_size = 1bp;

real time_length = 0.8;

//define coordinate unit vectors and points
triple I = (1, 0, 0);
triple J = (0, +1, 0);
triple K = (0, 0, +1);
triple x = (0, 0, 0);

//distance between CNN layers
real d = 4;

//distance between FC layers
real d_fc = 2;

//distance between channels in one CNN layer
real dx = 0.25;

//define the pen used for grid lines
pen grid_pen = grey;

//define the pen used for grid border
pen grid_border_pen = defaultpen;

//define the pen used for convolution kernels
pen conv_kernel_pen = linetype(new real[] {2,2}) + linewidth(0.3bp) + gray(0.1);

//define the pen used for convolution traces
pen conv_trace_pen = linetype(new real[] {2,2}) + linewidth(0.3bp) + gray(0.1);



//draw a face with particular gray value
// x: origin
// mu, nu: vectors spanning the face
// val: gray color value
void draw_face(triple x, triple mu, triple nu, real val)
{
	triple new_point = x;
	path3 p = x -- x + mu -- x + mu + nu -- x + nu -- cycle;
	draw(surface(p), gray(val), nolight);
	draw(p, gray(0.1), nolight);
}


//draw a grid with a pen
// sx: origin
// mu, nu: unit vectors spanning the unit face
// n_mu, n_nu: number of unit faces the grid should have in both directions
// links: draw link arrows?
// p: pen color of the filled surface?
void draw_grid(triple sx, triple mu, triple nu, int n_mu, int n_nu, bool links=false, pen p=grid_pen) {
	path3 boundary = sx -- sx + n_mu * mu -- sx + n_mu * mu + n_nu * nu -- sx + n_nu * nu -- cycle;
	draw(surface(boundary), p, nolight);

	for(int i_n = 0; i_n <= n_mu; ++i_n) {
		draw(sx + i_n * mu -- sx + i_n * mu + n_nu * nu, p=grey);
	}

	for(int j_n = 0; j_n <= n_nu; ++j_n) {
		draw(sx + j_n * nu -- sx + j_n * nu + n_mu * mu, p=grey);
	}

	pen lp = gray(0.3);
	if (links) {
		for(int i_n = 0; i_n < n_mu; ++i_n) {
			for(int j_n = +1; j_n < n_nu+1; ++j_n) {
				draw(sx + i_n * mu + j_n * nu -- sx + i_n * mu + j_n * nu  - nu, p=lp+solid, MidArcArrow3(size=arrow_size));
				draw(sx + i_n * mu + j_n * nu -- sx + i_n * mu + j_n * nu  + mu, p=lp+solid, MidArcArrow3(size=arrow_size));
			}
		}
	}

	//draw(boundary, black, nolight);
	draw(boundary, nolight, p=grid_border_pen);
}


//draw n channel grids in succession
//similar to draw_grid
//BUT
//c: CENTER of the grid (only odd sized usable!)
// n_ch: number of channels
void draw_n_ch_grid(triple c, triple mu, triple nu, int n_mu, int n_nu, int n_ch, pen p=gray(0.9))
{
  for (int i_ch = 0; i_ch < n_ch; ++i_ch)
  {
    draw_grid(c + i_ch*dx*J, mu, nu, n_mu, n_nu, p=p);
  }
}

//draw n channel grids in succession
//similar to draw_grid
//BUT
//c: CENTER of the grid (only odd sized usable!)
// n_ch: number of channels
void draw_n_ch_grid_kernel(triple c, triple mu, triple nu, int n_mu, int n_nu, int n_ch, triple kernel_c, int kernel_n_mu, int kernel_n_nu, pen kernel_pen=black, triple kernel_c_in=(0,0,0), int kernel_n_mu_in=0, int kernel_n_nu_in=0, pen kernel_pen_in=invisible)
{
  for (int i_ch = 0; i_ch < n_ch; ++i_ch)
  {
    draw_grid(c + i_ch*dx*J, mu, nu, n_mu, n_nu, grid_pen);
		draw_grid(kernel_c + i_ch*dx*J, mu, nu, kernel_n_mu, kernel_n_nu, p=kernel_pen);
		if (kernel_n_mu_in != 0 && kernel_n_nu_in != 0)
		{
		draw_grid(kernel_c_in + i_ch*dx*J, mu, nu, kernel_n_mu_in, kernel_n_nu_in, p=kernel_pen_in);
		}
	}
}

// draw a 2D convolution
// c: corner of the kernel depicted on the input lattice
// mu, nu: unit vectors spanning unit face
// n_mu, n_nu: size of the kernel in terms of unit faces
// t: target of the corner unit cell the kernel maps to on the output lattice
void draw_conv2d(triple c, triple mu0, triple nu0, int n_mu0, int n_nu0, int n_ch0, triple t, triple mu1, triple nu1, int n_mu1, int n_nu1, int n_ch1, pen kernel_pen=conv_kernel_pen)
{
  path3 kernel_in = c -- c + n_mu0 * mu0 -- c + n_mu0 * mu0 + n_nu0 * nu0 -- c + n_nu0 * nu0 -- cycle;
  path3 kernel_out = t -- t + n_mu1 * mu1 -- t + n_mu1 * mu1 + n_nu1 * nu1 -- t + n_nu1 * nu1 -- cycle;

  triple c_last = c + (n_ch0 - 1)*dx*J;

	draw(c_last + n_mu0 * mu0 -- t + n_mu1 * mu1, conv_trace_pen);
  draw(c_last + n_mu0 * mu0 + n_nu0 * nu0 -- t + n_mu1 * mu1 + n_nu1 * nu1, conv_trace_pen);
  draw(c_last + n_nu0 * nu0 -- t + n_nu1 * nu1, conv_trace_pen);
	draw(c_last -- t, conv_trace_pen);

}

triple center_to_corner(triple center, triple mu_vec, triple nu_vec)
{
  return center + -0.5*(mu_vec) + -0.5*(nu_vec);
}

//draw axes for reference

/*
Label x_axis_label = Label("$x$", position=Relative(1.0));
draw((0,0,0) -- 10*I, L=x_axis_label, EndArcArrow3(size=arrow_size));

//Label y_axis_label = Label("$y$", position=Relative(1.0));
//draw((0,0,0) -- 10*J, L=y_axis_label, EndArcArrow3(size=arrow_size));

Label z_axis_label = Label("$z$", position=Relative(1.0));
draw((0,0,0) -- 10*K, L=z_axis_label, EndArcArrow3(size=arrow_size));
*/

//generating an image

//define the layers

//first layer (input)
triple x0 = (0, 0, 0);
int n_mu0 = 8;
int n_nu0 = 8;
int n_ch0 = 1;
triple x0_corner = center_to_corner(x0, n_mu0*I, n_nu0*K);


//second layer
triple x1 = x0 + d * J;
int n_mu1 = 4;
int n_nu1 = 4;
int n_ch1 = 1;
triple x1_corner = center_to_corner(x1, n_mu1*I, n_nu1*K);

//third layer
triple x2 = x1 + d * J;
int n_mu2 = 2;
int n_nu2 = 2;
int n_ch2 = 1;
triple x2_corner = center_to_corner(x2, n_mu2*I, n_nu2*K);

//fourth layer (flattened layer)
triple x3 = x2 + d * J;
int n_mu3 = 1;
int n_nu3 = 4;
int n_ch3 = 1;
triple x3_corner = center_to_corner(x3, n_mu3*J, n_nu3*K);

//fifth layer (MLP)
//triple x4 = x3 + d_fc * J;
triple x4 = x3 + d * J;
int n_mu4 = 1;
int n_nu4 = 4;
int n_ch4 = 1;
triple x4_corner = center_to_corner(x4, n_mu4*J, n_nu4*K);

//output neuron
triple output = x4 + d_fc * J;
int n_mu_out = 1;
int n_nu_out = 1;
triple output_corner = center_to_corner(output, n_mu_out*J, n_nu_out*K);


//first kernel input
triple kernel_c0 = x0_corner + 2*I + 2*K;
int kernel_n_mu = 2;
int kernel_n_nu = 2;

//first kernel output
triple kernel_c0_out = x1_corner + I + K;

//second kernel input
triple kernel_c1 = x1_corner;
int kernel_n_mu = 2;
int kernel_n_nu = 2;

//second kernel output
triple kernel_c1_out = x2_corner;

//third kernel input
triple kernel_c2 = x2_corner;
int kernel_n_mu = 2;
int kernel_n_nu = 2;

//third kernel output
triple kernel_c2_out = x3_corner;


//draw the layers

//draw first layer (input) and kernel stencil
draw_n_ch_grid_kernel(x0_corner, I, K, n_mu0, n_nu0, n_ch0, kernel_c0, kernel_n_mu, kernel_n_nu, pink);

//draw first convolution trace
draw_conv2d(kernel_c0, I, K, kernel_n_mu, kernel_n_nu, n_ch0, kernel_c0_out, I, K, 1, 1, n_ch1, kernel_pen=pink);

//draw second layer and kernel stencils (in and out)
draw_n_ch_grid_kernel(x1_corner, I, K, n_mu1, n_nu1, n_ch1, kernel_c1, kernel_n_mu, kernel_n_nu, palecyan, kernel_c0_out, 1, 1, pink);

//draw second convolution trace
draw_conv2d(kernel_c1, I, K, kernel_n_mu, kernel_n_nu, n_ch1, kernel_c1_out, I, K, 1, 1, n_ch2, kernel_pen=palecyan);

//draw third layer and kernel stencils (in and out)
draw_n_ch_grid_kernel(x2_corner, I, K, n_mu2, n_nu2, n_ch2, kernel_c2, kernel_n_mu, kernel_n_nu, paleblue, kernel_c1_out, 1, 1, palecyan);


//draw third flattening trace
draw_conv2d(kernel_c2, I, K, kernel_n_mu, kernel_n_nu, n_ch2, kernel_c2_out, I, K, n_mu3, n_nu3, 1, kernel_pen=palecyan);

//draw fourth flattened layer and input stencils
draw_n_ch_grid_kernel(x3_corner, I, K, n_mu3, n_nu3, n_ch3, x3_corner, n_mu3, n_nu3, paleblue);

//track flattened variable squares
draw_grid(x3_corner, I, K, 1, 1, p=palecyan);
//draw_grid(x3_corner + 4*K, I, K, 1, 1, p=pink);

//first connected layer
draw_conv2d(x3_corner, I, K, n_mu3, n_nu3, 1, x4_corner, I, K, n_mu4, n_nu4, 1, kernel_pen=paleblue);

//draw fourth flattened layer and input stencils
draw_n_ch_grid_kernel(x4_corner, I, K, n_mu4, n_nu4, n_ch4, x4_corner, n_mu4, n_nu4, paleblue);

//second connected layer (to output)
draw_conv2d(x4_corner, I, K, n_mu4, n_nu4, 1, output_corner, I, K, n_mu_out, n_nu_out, 1, kernel_pen=paleblue);

//output neuron
draw_n_ch_grid_kernel(output_corner, I, K, 1, 1, 1, output_corner, 1, 1, paleblue);


//labels

pen label_pen = fontsize(7pt);

//label the first layer
Label x0_label = rotate(angle=-90, u=(0,0,0), v=(1,0,0))*Label("Input", p=label_pen);
Label x0_label = Label("Input", p=label_pen);
label(x0_label, x0 + n_nu0/2*K + 2.5*K + -2*J);

//label the conv layers
Label x2_label = rotate(angle=-90, u=(0,0,0), v=(1,0,0))*Label("Conv2D", p=label_pen);
Label x2_label = Label("Convolutions/pooling", p=label_pen);
label(x2_label, (x1+x2)/2 + (n_nu0/2 + 0)*K);

Label stride = Label("$(\text{stride}>1)\mspace{2mu}$\ding{55}", label_pen + red);
label(stride, (x1+x2)/2 + (n_nu0/2 + 1.)*K);

//label the fourth layer (flattening)
Label x3_label = rotate(angle=-90, u=(0,0,0), v=(1,0,0))*Label("Flatten", p=label_pen);
Label x3_label = Label("Flattening$\mspace{2mu}$\ding{55}", p=label_pen + red);
//label(x3_label, x3 + 1*(n_nu0/2)*K);
label(x3_label, x3 + -1*(n_nu0/2)*K);

//label the fifth layer
Label x4_label = rotate(angle=-90, u=(0,0,0), v=(1,0,0))*Label("MLP", p=label_pen);
Label x4_label = Label("\centering Dense \par network", p=label_pen);
label(x4_label, x4 + (n_nu0/2)*K - 1.0*K);

//label the sixth layer
Label x4_label = rotate(angle=-90, u=(0,0,0), v=(1,0,0))*Label("Output", p=label_pen);
Label x4_label = Label("Output", p=label_pen);
label(x4_label, output + 2.5*J);


//overall label
Label schem_label = Label("\centering Flattening \par architecture \par (FL)", p=label_pen);
//label(schem_label, output + n_mu0*I - n_mu0*K);
label(schem_label, output - n_mu0*K + 10*dx*J);
label(schem_label, output + 2.5*J - n_mu0*K);
