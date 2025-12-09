package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"image/color"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// ПАРАМЕТРЫ ГА
const (
	PopSize        = 300  // размер популяции
	Generations    = 1000 // число поколений
	EliteCount     = 10   // сколько лучших переносим без изменений
	CrossoverRate  = 0.9  // доля пар, у которых делаем кроссовер
	MutationRate   = 0.2  // вероятность мутации гена
	MutationSigma  = 2.0  // амплитуда мутации
	TournamentSize = 5    // турнирный отбор

)

type State struct {
	X1, X2, X3 float64
}

// особь ГА
type Individual struct {
	A []float64 // узлы для u1
	B []float64 // узлы для u2
	J float64   // значение функционала
}

// для отрисовки
type Point struct{ X, Y float64 }

// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func lerp(v0, v1, t0, t1, t float64) float64 {
	if t1 == t0 {
		return v0
	}
	return v0 + (v1-v0)*(t-t0)/(t1-t0)
}

// ПАРАМЕТРЫ ЗУ
const (
	TMax      = 15.0 // максимальное время моделирования
	NSegments = 20   // 20 сегментов управления
	GoalEps   = 0.20

	Umin = -10.0
	Umax = 10.0

	DT = 0.02

	AlphaPen     = 8000.0    // штраф в зоне безопасности
	CollisionPen = 1000000.0 // огромный штраф за заход внутрь круга
	SafeMargin   = 0.010     // очень узкая зона безопасности

	// вес за энергию управления
	EnergyWeight = 0.0010

	// гладкость
	SmoothWeight = 0.10

	// вес расстояния до цели в момент tf
	GoalWeight = 80.0
)

func controlAt(ind *Individual, t float64) (u1, u2 float64) {
	delta := TMax / float64(NSegments)
	k := int(math.Floor(t / delta))
	if k < 0 {
		k = 0
	}
	if k >= NSegments {
		k = NSegments - 1
	}
	t0 := float64(k) * delta
	t1 := float64(k+1) * delta

	a0 := ind.A[k]
	a1 := ind.A[k+1]
	b0 := ind.B[k]
	b1 := ind.B[k+1]

	u1 = clamp(lerp(a0, a1, t0, t1, t), Umin, Umax)
	u2 = clamp(lerp(b0, b1, t0, t1, t), Umin, Umax)
	return
}

func dynamics(x State, u1, u2 float64) State {
	s := 0.5 * (u1 + u2)
	w := 0.5 * (u1 - u2)
	return State{
		X1: s * math.Cos(x.X3),
		X2: s * math.Sin(x.X3),
		X3: w,
	}
}

func rk4Step(x State, u1, u2, h float64) State {
	k1 := dynamics(x, u1, u2)

	x2 := State{x.X1 + 0.5*h*k1.X1, x.X2 + 0.5*h*k1.X2, x.X3 + 0.5*h*k1.X3}
	k2 := dynamics(x2, u1, u2)

	x3 := State{x.X1 + 0.5*h*k2.X1, x.X2 + 0.5*h*k2.X2, x.X3 + 0.5*h*k2.X3}
	k3 := dynamics(x3, u1, u2)

	x4 := State{x.X1 + h*k3.X1, x.X2 + h*k3.X2, x.X3 + h*k3.X3}
	k4 := dynamics(x4, u1, u2)

	return State{
		X1: x.X1 + (h/6.0)*(k1.X1+2*k2.X1+2*k3.X1+k4.X1),
		X2: x.X2 + (h/6.0)*(k1.X2+2*k2.X2+2*k3.X2+k4.X2),
		X3: x.X3 + (h/6.0)*(k1.X3+2*k2.X3+2*k3.X3+k4.X3),
	}
}

func h1(x State) float64 {
	return 2.5 - math.Hypot(x.X1-2.5, x.X2-2.5)
}
func h2(x State) float64 {
	return 2.5 - math.Hypot(x.X1-7.5, x.X2-7.5)
}

// штраф за препятствия
func phi(h float64) float64 {

	if h > 0.0 {
		return CollisionPen + AlphaPen*h*h
	}

	if h > -SafeMargin {
		d := h + SafeMargin
		return AlphaPen * d * d
	}

	return 0.0
}

func goalDist(x State) float64 {
	return math.Hypot(x.X1, x.X2)
}

func smoothnessCost(ind *Individual) float64 {
	var sum float64
	for i := 0; i < len(ind.A)-1; i++ {
		da := ind.A[i+1] - ind.A[i]
		db := ind.B[i+1] - ind.B[i]
		sum += da*da + db*db
	}
	return SmoothWeight * sum
}

func simulateAndCost(ind *Individual) (J float64, tFinal float64, reached bool) {
	x := State{X1: 10.0, X2: 10.0, X3: 0.0}

	var penaltyInt float64
	var energyInt float64
	var t float64

	steps := int(TMax / DT)
	for i := 0; i < steps; i++ {
		u1, u2 := controlAt(ind, t)

		energyInt += (u1*u1 + u2*u2) * DT

		x = rk4Step(x, u1, u2, DT)

		penaltyInt += (phi(h1(x)) + phi(h2(x))) * DT

		t += DT

		if goalDist(x) <= GoalEps {
			tFinal = t
			reached = true
			break
		}
	}

	if !reached {
		tFinal = TMax
	}

	term := goalDist(x) * GoalWeight

	smooth := smoothnessCost(ind)

	energyCost := EnergyWeight * energyInt

	J = tFinal + term + penaltyInt + smooth + energyCost

	if !reached {
		J += 1000.0
	}

	return
}

func simulateAndCostWithPath(ind *Individual) (J float64, tFinal float64, reached bool, path []Point) {
	x := State{X1: 10.0, X2: 10.0, X3: 0.0}

	var penaltyInt float64
	var energyInt float64
	var t float64

	path = append(path, Point{X: x.X1, Y: x.X2})

	steps := int(TMax / DT)
	for i := 0; i < steps; i++ {
		u1, u2 := controlAt(ind, t)

		energyInt += (u1*u1 + u2*u2) * DT

		x = rk4Step(x, u1, u2, DT)

		penaltyInt += (phi(h1(x)) + phi(h2(x))) * DT

		t += DT

		path = append(path, Point{X: x.X1, Y: x.X2})

		if goalDist(x) <= GoalEps {
			tFinal = t
			reached = true
			break
		}
	}

	if !reached {
		tFinal = TMax
	}

	term := goalDist(x) * GoalWeight
	smooth := smoothnessCost(ind)
	energyCost := EnergyWeight * energyInt

	J = tFinal + term + penaltyInt + smooth + energyCost
	if reached {
		path = append(path, Point{X: 0.0, Y: 0.0})
	}

	return
}

func circlePolygon(cx, cy, r float64, n int) plotter.XYs {
	pts := make(plotter.XYs, n+1)
	for i := 0; i <= n; i++ {
		ang := 2 * math.Pi * float64(i) / float64(n)
		pts[i].X = cx + r*math.Cos(ang)
		pts[i].Y = cy + r*math.Sin(ang)
	}
	return pts
}

func boundsWithPadding(path []Point, padFrac float64) (xmin, xmax, ymin, ymax float64) {
	if len(path) == 0 {
		return -1, 11, -1, 11
	}
	xmin, xmax = path[0].X, path[0].X
	ymin, ymax = path[0].Y, path[0].Y
	for i := 1; i < len(path); i++ {
		if path[i].X < xmin {
			xmin = path[i].X
		}
		if path[i].X > xmax {
			xmax = path[i].X
		}
		if path[i].Y < ymin {
			ymin = path[i].Y
		}
		if path[i].Y > ymax {
			ymax = path[i].Y
		}
	}

	circles := [][3]float64{{2.5, 2.5, 2.5}, {7.5, 7.5, 2.5}}
	for _, c := range circles {
		cx, cy, r := c[0], c[1], c[2]
		if cx-r < xmin {
			xmin = cx - r
		}
		if cx+r > xmax {
			xmax = cx + r
		}
		if cy-r < ymin {
			ymin = cy - r
		}
		if cy+r > ymax {
			ymax = cy + r
		}
	}

	w := math.Max(1e-9, xmax-xmin)
	h := math.Max(1e-9, ymax-ymin)
	px := padFrac * w
	py := padFrac * h
	return xmin - px, xmax + px, ymin - py, ymax + py
}

func saveTrajectoryPNG(filename string, path []Point) error {

	xy := make(plotter.XYs, len(path))
	for i := range path {
		xy[i].X = path[i].X
		xy[i].Y = path[i].Y
	}

	p := plot.New()
	p.Title.Text = "Траектория движения робота"
	p.X.Label.Text = "x1 (ось X)"
	p.Y.Label.Text = "x2 (ось Y)"
	p.Add(plotter.NewGrid())

	xmin, xmax, ymin, ymax := boundsWithPadding(path, 0.08)
	p.X.Min, p.X.Max = xmin, xmax
	p.Y.Min, p.Y.Max = ymin, ymax

	poly1, err := plotter.NewPolygon(circlePolygon(2.5, 2.5, 2.5, 160))
	if err != nil {
		return err
	}
	poly1.Color = color.RGBA{128, 128, 128, 80}
	poly1.LineStyle.Width = 0
	p.Add(poly1)

	poly2, err := plotter.NewPolygon(circlePolygon(7.5, 7.5, 2.5, 160))
	if err != nil {
		return err
	}
	poly2.Color = color.RGBA{128, 128, 128, 80}
	poly2.LineStyle.Width = 0
	p.Add(poly2)

	line, _ := plotter.NewLine(xy)
	line.Color = color.RGBA{0, 80, 255, 255}
	line.Width = vg.Points(1.8)
	p.Add(line)
	p.Legend.Add("Траектория", line)

	startPts := plotter.XYs{{X: 10.0, Y: 10.0}}
	start, _ := plotter.NewScatter(startPts)
	start.GlyphStyle.Shape = draw.CircleGlyph{}
	start.GlyphStyle.Color = color.RGBA{0, 140, 0, 255}
	start.GlyphStyle.Radius = vg.Points(4)
	p.Add(start)
	p.Legend.Add("Начало", start)

	goalPts := plotter.XYs{{X: 0.0, Y: 0.0}}
	goal, _ := plotter.NewScatter(goalPts)
	goal.GlyphStyle.Shape = draw.CrossGlyph{}
	goal.GlyphStyle.Color = color.RGBA{220, 0, 0, 255}
	goal.GlyphStyle.Radius = vg.Points(4)
	p.Add(goal)
	p.Legend.Add("Цель", goal)

	p.Legend.Add("Препятствие 1", poly1)
	p.Legend.Add("Препятствие 2", poly2)

	return p.Save(6*vg.Inch, 6*vg.Inch, filename)
}

// создание случайной особи
func newRandomIndividual() *Individual {
	a := make([]float64, NSegments+1)
	b := make([]float64, NSegments+1)

	const initStep = 2.0

	a[0] = Umin + rand.Float64()*(Umax-Umin)
	b[0] = Umin + rand.Float64()*(Umax-Umin)

	for i := 1; i < len(a); i++ {
		da := (rand.Float64()*2 - 1) * initStep
		db := (rand.Float64()*2 - 1) * initStep
		a[i] = clamp(a[i-1]+da, Umin, Umax)
		b[i] = clamp(b[i-1]+db, Umin, Umax)
	}

	return &Individual{A: a, B: b, J: math.Inf(1)}
}

// копия
func clone(ind *Individual) *Individual {
	a := make([]float64, len(ind.A))
	b := make([]float64, len(ind.B))
	copy(a, ind.A)
	copy(b, ind.B)
	return &Individual{A: a, B: b, J: ind.J}
}

func evaluate(ind *Individual) {
	J, _, _ := simulateAndCost(ind)
	ind.J = J
}

func tournament(pop []*Individual) *Individual {
	best := pop[rand.Intn(len(pop))]
	for i := 1; i < TournamentSize; i++ {
		cand := pop[rand.Intn(len(pop))]
		if cand.J < best.J {
			best = cand
		}
	}
	return best
}

func crossover(p1, p2 *Individual) (*Individual, *Individual) {
	c1 := clone(p1)
	c2 := clone(p2)
	if rand.Float64() > CrossoverRate {
		return c1, c2
	}
	lambda := rand.Float64()
	for i := range c1.A {
		c1.A[i] = clamp(lambda*p1.A[i]+(1-lambda)*p2.A[i], Umin, Umax)
		c2.A[i] = clamp(lambda*p2.A[i]+(1-lambda)*p1.A[i], Umin, Umax)
	}
	for i := range c1.B {
		c1.B[i] = clamp(lambda*p1.B[i]+(1-lambda)*p2.B[i], Umin, Umax)
		c2.B[i] = clamp(lambda*p2.B[i]+(1-lambda)*p1.B[i], Umin, Umax)
	}
	return c1, c2
}

func mutate(ind *Individual) {
	for i := range ind.A {
		if rand.Float64() < MutationRate {
			ind.A[i] = clamp(ind.A[i]+(rand.Float64()*2-1)*MutationSigma, Umin, Umax)
		}
	}
	for i := range ind.B {
		if rand.Float64() < MutationRate {
			ind.B[i] = clamp(ind.B[i]+(rand.Float64()*2-1)*MutationSigma, Umin, Umax)
		}
	}
}

func nextGeneration(pop []*Individual) []*Individual {

	sortByCost(pop)

	newPop := make([]*Individual, 0, len(pop))
	for i := 0; i < EliteCount; i++ {
		newPop = append(newPop, clone(pop[i]))
	}

	for len(newPop) < len(pop) {
		p1 := tournament(pop)
		p2 := tournament(pop)
		c1, c2 := crossover(p1, p2)
		mutate(c1)
		mutate(c2)
		evaluate(c1)
		if len(newPop) < len(pop) {
			newPop = append(newPop, c1)
		}
		if len(newPop) < len(pop) {
			evaluate(c2)
			newPop = append(newPop, c2)
		}
	}
	return newPop
}

func sortByCost(pop []*Individual) {

	for i := 1; i < len(pop); i++ {
		j := i
		for j > 0 && pop[j].J < pop[j-1].J {
			pop[j], pop[j-1] = pop[j-1], pop[j]
			j--
		}
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())

	pop := make([]*Individual, PopSize)
	for i := range pop {
		pop[i] = newRandomIndividual()
		evaluate(pop[i])
	}

	best := pop[0]
	for g := 0; g < Generations; g++ {
		pop = nextGeneration(pop)

		sortByCost(pop)
		if pop[0].J < best.J {
			best = clone(pop[0])
		}
		if g%10 == 0 || g == Generations-1 {
			log.Printf("Gen %3d | best J = %.5f", g, pop[0].J)
		}
	}

	J, tf, reached, path := simulateAndCostWithPath(best)
	fmt.Println("------------ RESULT ------------")
	fmt.Printf("Best J:       %.6f\n", J)
	fmt.Printf("Reached goal: %v\n", reached)
	dist := 0.0
	if len(path) > 0 {
		last := path[len(path)-1]
		dist = math.Sqrt(last.X*last.X + last.Y*last.Y)
	}
	fmt.Printf("Final Dist:   %.6f\n", dist)
	fmt.Printf("t_final:      %.3f s\n", tf)

	// PNG
	if err := saveTrajectoryPNG("trajectory.png", path); err != nil {
		log.Fatalf("plot error: %v", err)
	}
	fmt.Println("PNG сохранён: trajectory.png")

	fmt.Println("\nU1 nodes (a_k):")
	for i, v := range best.A {
		fmt.Printf("k=%02d  a=%8.4f\n", i, v)
	}
	fmt.Println("\nU2 nodes (b_k):")
	for i, v := range best.B {
		fmt.Printf("k=%02d  b=%8.4f\n", i, v)
	}

	fmt.Println("\nSample controls u1(t), u2(t):")
	delta := TMax / float64(NSegments)
	for k := 0; k <= NSegments; k++ {
		t := float64(k) * delta
		u1, u2 := controlAt(best, t)
		fmt.Printf("t=%6.2f  u1=%8.4f  u2=%8.4f\n", t, u1, u2)
	}

}
