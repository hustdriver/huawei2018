#include "predict.h"
#include <stdio.h>
#include <iostream>
using std::cout; using std::cin; using std::endl; using std::cerr;
#include <sstream>
using std::istringstream; using std::ostringstream;
#include <string>
using std::string;
using std::to_string;
#include <vector>
using std::vector;
#include <tuple>
using std::tuple; using std::get;
#include <cstddef>  // for size_t
#include <cstring>  // for strcmp
#include <utility>
using std::pair;
#include <cmath>
#include <map>
using std::map;
#include<algorithm>

#define TOTAL_FLAVOR_TYPE_NUM 18


int sum_vm = 0;     // The sum of virtual machines
int average = 9;
int leijia = 0;//AR模型需要
int AR_DAYS = 25;//40
int LR_DAYS = 25;


double **Transpose(double **matrix_, int M, int N);
double **Multiply(double**matrix_x, double**matrix_y, int X, int Y, int Z);
void LUP_Descomposition(double **A_temp, double **L, double **U, int *P, int N);
double * LUP_Solve(double **L, double **U, int *P, double *b, int N);
double ** inverse(double** A, int N);
double ** inverse(double** A, int N);
vector<double> calculateA(vector<double>in, int p);
double piancha(vector<double>in, int p, vector<double>A);

struct Flavor {
	string name;
	int id;
	double cpudemand;
	double memdemand;
	int num = 0;    // The number of each flavor vm
};
struct plan
{
	double cpusum = 0;
	double memsum = 0;
	vector<Flavor>placeplan;//这个计划放置虚拟机的顺序
};
struct Server
{
	int type;
	int num = 0;//该类型虚拟机使用数量
	string name;
	double cpucap;
	double memcap;
	double harddiskcap;
	bool isuse = false;
	vector<plan> distplan;//num条放置的计划
};


int day_of_year(int, int, int);
int day_of_year(string /* date format of '2015-02-03' */);
typedef tuple<vector<vector<int>>, string, string> DataFormat;
typedef tuple<map<string, vector<int>>, int, vector<Flavor>, string, int> InfoFormat;

DataFormat processdata(char **, int);
InfoFormat processinfo(char **);

static char daytab[2][13] = {
	{ 0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
	{ 0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
};


int day_of_year(int year, int month, int day) {
	int leap;

	leap = year % 4 == 0 && year % 100 != 0 || year % 400 == 0;
	for (int i = 1; i < month; i++)
		day += daytab[leap][i];
	return day;
}

int day_of_year(string date) {
	int year = stoi(date.substr(0, 4));
	int month = stoi(date.substr(5, 2));
	int day = stoi(date.substr(8, 2));
	return day_of_year(year, month, day);
}

int rdn(int y, int m, int d) { /* Rata Die day one is 0001-01-01 */
	if (m < 3)
		y--, m += 12;
	return 365 * y + y / 4 - y / 100 + y / 400 + (153 * m - 457) / 5 + d - 306;
}

int day_relative_to_initialdate(string date, string initialdate) {
	int year = stoi(date.substr(0, 4));
	int month = stoi(date.substr(5, 2));
	int day = stoi(date.substr(8, 2));

	int initialyear = stoi(initialdate.substr(0, 4));
	int initialmonth = stoi(initialdate.substr(5, 2));
	int initialday = stoi(initialdate.substr(8, 2));
	return rdn(year, month, day) - rdn(initialyear, initialmonth, initialday);
}

void print_array(int a) {
	cout << a << " ";
}

template <class T>
void print_array(const T& arr) {
	for (const auto v : arr) {
		print_array(v);
	}
	cout << endl;
}



double sum_p(vector<double>::iterator a, vector<double>::iterator b, int n)
{
	double sum = 0;
	for (int j = 0; j < n; j++)
		sum += ((double)*(a + j)) * (*(b + j));
	return sum;
}

void evaluate(int m, vector<double>dataArray, double &accu, vector<double> w)
{
	vector<double> pre(dataArray.size() - m, 0);

	accu = 0;
	for (unsigned int i = 0; i < pre.size(); i++)
	{
		pre[i] = sum_p(dataArray.begin() + i, w.begin(), m);
		accu += (pre[i] - dataArray[i + m])*(pre[i] - dataArray[i + m]);
	}

}

vector<double> predict_history(vector<vector<int>>data, int mod)
{
	double ave = 0;
	for (int i = 0; i < data.size(); i++)
	{
		ave += data[i][mod];
	}
	ave = ave / data.size();
	double o = 0;
	for (int i = 0; i < data.size(); i++)
	{
		o += (data[i][mod] - ave)*(data[i][mod] - ave);
	}
	o = o / data.size();
	o = sqrt(o);

	vector<double> in;
	int l = data.size();
	for (int i = data.size(); i > 0; i--)//前parameter天为预测向量
	{
		if (abs(data[l - i][mod] - ave) <= 10 * o)
		{
			in.push_back(data[l - i][mod]);
		}
	}
	return in;
}

int predictzhishu(vector<double>dataArray, int n)
{
	int sum = 0;
	vector<double>pt(n, 0);
	int n1 = 4;//需评估的平滑系数个数
	int m = 15;//与预测值有关的历史值个数
	//平滑系数集合
	vector<double> a(n1, 0.2);
	for (int i = 1; i < n1; i++)
		a[i] = a[i - 1] / 2;

	//记录平滑系数对应的预测精度
	vector<double> accu(n1);
	//计算平滑系数对应的各数权重
	vector<vector<double> > w(n1, vector<double>(m));
	for (int i = 0; i < n1; i++)
	{
		w[i][m - 1] = a[i];
		for (int j = m - 2; j > 0; j--)
			w[i][j] = w[i][j + 1] * (1 - a[i]);
		w[i][0] = 1;
		for (int j = m - 2; j >= 0; j--)
			w[i][0] *= (1 - a[i]);
	}
	//评估并确定最优的平滑系数ao对应的编号index/预测精度accuo
	for (int i = 0; i < n1; i++)
	{
		evaluate(m, dataArray, accu[i], w[i]);
	}
	double accuo = accu[0];
	int index = 0;
	for (int i = 1; i < n1; i++)
		if (accu[i] < accuo)
		{
			accuo = accu[i];
			index = i;
		};


	cout << "预测精度:";
	for (int i = 0; i < n1; i++)
		cout << accu[i] << '\t';
	cout << endl;
	cout << "最佳预测精度:" << accu[index];
	cout << endl;

	//获得预测值
	vector<double> tempArray(m);
	tempArray.assign(dataArray.end() - m, dataArray.end());

	for (int i = 0; i < n; i++)
	{
		pt[i] = ceil(sum_p(tempArray.begin(), w[index].begin(), m));
		tempArray.erase(tempArray.begin());
		tempArray.push_back(pt[i]);
		sum += pt[i];
		/*for (unsigned int j = 0; j < tempArray.size() - 1; j++)
		{
		tempArray[j] = tempArray[j + 1];
		}
		tempArray[tempArray.size() - 1] = pt[i];*/
	}

	return sum;
}


//不去噪声
/*
vector<double> predict_in(vector<vector<int>>data, int mod, int parameter)
{
	vector<double> in;
	int l = data.size();
	int m = 1;
	for (int i = parameter; i > 0; i--)//前parameter天为预测向量
	{
		if (data[l - i][mod] <= noise)
		{
			in.push_back(data[l - i][mod]+leijia);
		}
		else if (data[l - i][mod] > noise)
		{
			while (data[l - parameter - m][mod] > noise)
			{
				m++;
			}
			in.insert(in.begin(), data[l - parameter - m][mod]);
		}
	}
	return in;
}*/

void LR(vector<double>data_x, vector<double>data_y, int data_n, vector<double> &vResult)
{
	double A = 0.0;
	double B = 0.0;
	double C = 0.0;
	double D = 0.0;
	double E = 0.0;
	double F = 0.0;

	for (int i = 0; i < data_n; i++)
	{
		A += data_x[i] * data_x[i];
		B += data_x[i];
		C += data_x[i] * data_y[i];
		D += data_y[i];
	}

	// 计算斜率a和截距b  
	double a, b, temp = 0;
	if (temp = (data_n*A - B*B))// 判断分母不为0  
	{
		a = (data_n*C - B*D) / temp;
		b = (A*D - B*C) / temp;
	}
	else
	{
		a = 1;
		b = 0;
	}
	// 计算相关系数r  
	double Xmean, Ymean;
	Xmean = B / data_n;
	Ymean = D / data_n;

	double tempSumXX = 0.0, tempSumYY = 0.0;
	for (int i = 0; i < data_n; i++)
	{
		tempSumXX += (data_x[i] - Xmean) * (data_x[i] - Xmean);
		tempSumYY += (data_y[i] - Ymean) * (data_y[i] - Ymean);
		E += (data_x[i] - Xmean) * (data_y[i] - Ymean);
	}
	F = sqrt(tempSumXX) * sqrt(tempSumYY);

	double r;
	r = E / F;

	vResult.push_back(a);
	vResult.push_back(b);
	vResult.push_back(r*r);
}

/*vector<double> predict_in(vector<vector<int>>data, int mod, int parameter)
{
	vector<double> in;
	int l = data.size();
	int m = 1;
	for (int i = parameter; i > 0; i--)//前parameter天为预测向量
	{
		if (data[l - i][mod] <= noise)
		{
			in.push_back(data[l - i][mod] + leijia);
		}
		else if (data[l - i][mod] > noise)
		{
			while (data[l - parameter - m][mod] > noise)
			{
				m++;
			}
			in.insert(in.begin(), data[l - parameter - m][mod] + leijia);
		}
	}
	return in;
}
*/

vector<double> predict_in(vector<vector<int>>data, int mod, int parameter)
{
	double ave = 0;
	for (int i = 0; i < data.size(); i++)
	{
		ave += data[i][mod];
	}
	ave = ave / data.size();
	double o = 0;
	for (int i = 0; i < data.size(); i++)
	{
		o += (data[i][mod] - ave)*(data[i][mod] - ave);
	}
	o = o / data.size();
	o = sqrt(o);

	vector<double> in;
	int l = data.size();
	int m = 1;
	for (int i = parameter + 1; i > 0; i--)//前parameter天为预测向量
	{
		if (abs(data[l - i][mod] - ave) <= 3 * o)
		{
			in.push_back(data[l - i][mod]);
		}
		else if (data[l - i][mod] > 3 * o)
		{
			while (abs(data[l - parameter - m][mod] - ave) > 3 * o)
			{
				m++;
			}
			in.insert(in.begin(), data[l - parameter - m][mod]);
		}
	}
	return in;
}

double predict_next(vector<double>in, int mod)
{
	double sum = 0;
	for (int i = 0; i <in.size(); i++)
	{
		sum += (in[i]);
	}
	return (sum / in.size());
}

int predict_AR(vector<vector<int>>data, int days, int mod, int parameter,int freedayspan)
{
	vector<double> in = predict_in(data, mod, parameter);
	int l = data.size();
	double sum = 0.0;
	vector<double>Acoe;
	double minerror = 99999;
	int bestp;
	for (int m = 1; m < in.size() / 2; m++)
	{
		vector<double>tempA = calculateA(in, m);
		double errortemp = piancha(in, m, tempA);
		if (minerror > errortemp)
		{
			minerror = errortemp;
			bestp = m;
			Acoe = tempA;
		}
	}
	int p = bestp;

	for (int i = 0; i < freedayspan; i++)
	{
		double	next = 0.0;
		for (int j = 0; j < p; j++)
		{
			next += Acoe[j] * in[in.size() + j - p] + minerror*(rand() % 100 - 50.0) / 100;
		}
		double suiji = (rand() % 100 - 50.0);
		in.erase(in.begin());
		in.push_back(next);
	}
	for (int i = 0; i < days; i++)
	{
		double	next = 0.0;
		for (int j = 0; j < p; j++)
		{
			next += Acoe[j] * in[in.size() + j - p] + minerror*(rand() % 100 - 50.0) / 100;
		}
		double suiji = (rand() % 100 - 50.0);
		in.erase(in.begin());
		in.push_back(next);
		sum += (round(next - leijia));
	}
	return abs(round(sum));
}

int predict_LR(vector<vector<int>>data, int days, int mod, int parameter, int freedayspan)
{
	vector<double> in = predict_in(data, mod, parameter);
	int l = data.size();
	float	next, sum = 0;
	vector<double>coe;//存放线性回归系数
	vector<double>px;
	for (int m = 1; m <= parameter; m++)
	{
		px.push_back(m);
	}
	LR(px, in, parameter, coe);
	float a = coe[0], b = coe[1];
	for (int i = 0; i < days; i++)
	{
		next = a*(parameter + freedayspan + i + 1) + b;
		sum += next;
	}
	return abs(round(sum));
}

int predict_sum(vector<vector<int>>data, int days, int mod, int parameter, int freedayspan)
{
	vector<double> in = predict_in(data, mod, parameter);
	int l = data.size();
	double	next, sum = 0;
	for (int j = 0; j < freedayspan; j++)
	{
		next = predict_next(in, mod);
		in.erase(in.begin());
		in.push_back(next);
	}

	for (int i = 0; i < days; i++)
	{
		next = predict_next(in, mod);
		sum += next;
		in.erase(in.begin());
		in.push_back(next);
	}
	return (ceil(sum));
}


/*plan search_max(Server ps, vector<Flavor>needed_palce)
{
	//random_shuffle(needed_palce.begin(), needed_palce.end(),myrandom);//打乱排序

	vector<plan> pack;
	for (int i = 0; i < ps.cpucap + 1; i++)
	{
		plan tempplan;
		pack.push_back(tempplan);
	}
	for (int i = needed_palce.size() - 1; i >= 0; i--)//n个物品循环
		//for (int i = 0; i < needed_palce.size() ; i++)//n个物品循环
	{
		for (int j = ps.cpucap; j >= needed_palce[i].cpudemand; j--)//剩余容量
		{
			if (pack[j].cpusum == ps.cpucap&&pack[j].memsum == ps.memcap)
			{
				return pack[j];
			}
			if (pack[j].cpusum + needed_palce[i].cpudemand <= j
				&&pack[j].memsum + needed_palce[i].memdemand <= ps.memcap)
			{
				pack[j].cpusum += needed_palce[i].cpudemand;
				pack[j].memsum += needed_palce[i].memdemand;
				pack[j].placeplan.push_back(needed_palce[i]);
			}
			if (((double)pack[j].cpusum / ps.cpucap + pack[j].memsum / ps.memcap) <
				((pack[j - needed_palce[i].cpudemand].cpusum + needed_palce[i].cpudemand) / ps.cpucap +
				(pack[j - needed_palce[i].cpudemand].memsum + needed_palce[i].memdemand) / ps.memcap)
				&& (pack[j - needed_palce[i].cpudemand].cpusum + needed_palce[i].cpudemand) <= j - needed_palce[i].cpudemand
				&& (pack[j - needed_palce[i].cpudemand].memsum + needed_palce[i].memdemand) <= ps.memcap)
			{
				pack[j].placeplan = pack[j - needed_palce[i].cpudemand].placeplan;
				pack[j].placeplan.push_back(needed_palce[i]);
				pack[j].cpusum = pack[j - needed_palce[i].cpudemand].cpusum + needed_palce[i].cpudemand;
				pack[j].memsum = pack[j - needed_palce[i].cpudemand].memsum + needed_palce[i].memdemand;

			}
		}
	}
	return pack[ps.cpucap];
}
*/

/*plan search_max(Server ps, vector<Flavor>needed_palce)
{
	plan tempplan;
	for (int i = 0; i < needed_palce.size(); i++)
	{
		if ((tempplan.cpusum + needed_palce[i].cpudemand <= ps.cpucap) &&
			(tempplan.memsum + needed_palce[i].memdemand <= ps.memcap))
		{
			tempplan.cpusum += needed_palce[i].cpudemand;
			tempplan.memsum += needed_palce[i].memdemand;
			tempplan.placeplan.push_back(needed_palce[i]);
		}
	}
	return tempplan;
}*/

plan search_max(Server ps, vector<Flavor>needed_palce)
{
	plan tempplan;
	vector<Flavor>temp = needed_palce;
	/*tempplan.cpusum += temp[0].cpudemand;
	tempplan.memsum += temp[0].memdemand;
	tempplan.placeplan.push_back(temp[0]);
	temp.erase(temp.begin());*/
	if (temp.size() == 0)
	{
		return tempplan;
	}
	double bili = (ps.memcap - tempplan.memsum) / (ps.cpucap - tempplan.cpusum);
	for (int i = 0; i < temp.size() + 1; i++)
	{
		if (i == temp.size())
		{
			if ((tempplan.cpusum + temp[i - 1].cpudemand <= ps.cpucap) &&
				(tempplan.memsum + temp[i - 1].memdemand <= ps.memcap))
			{
				tempplan.cpusum += temp[i - 1].cpudemand;
				tempplan.memsum += temp[i - 1].memdemand;
				tempplan.placeplan.push_back(temp[i - 1]);
				temp.erase(temp.end() - 1);
				if (temp.size() == 0)
				{
					return tempplan;
				}
				else
					i = 0;
				bili = (ps.memcap - tempplan.memsum) / (ps.cpucap - tempplan.cpusum);
			}
			else
				return tempplan;

		}


		if (abs(bili - 1) <= abs(bili - 2) && abs(bili - 1) <= abs(bili - 4))
		{
			if ((tempplan.cpusum + temp[i].cpudemand <= ps.cpucap) &&
				(tempplan.memsum + temp[i].memdemand <= ps.memcap) &&
				(temp[i].memdemand / temp[i].cpudemand == 1))
			{
				tempplan.cpusum += temp[i].cpudemand;
				tempplan.memsum += temp[i].memdemand;
				tempplan.placeplan.push_back(temp[i]);
				temp.erase(temp.begin() + i);
				i = 0;
				bili = (ps.memcap - tempplan.memsum) / (ps.cpucap - tempplan.cpusum);
			}
		}
		else if (abs(bili - 2) <= abs(bili - 1) && abs(bili - 2) <= abs(bili - 4))
		{
			if ((tempplan.cpusum + temp[i].cpudemand <= ps.cpucap) &&
				(tempplan.memsum + temp[i].memdemand <= ps.memcap) &&
				(temp[i].memdemand / temp[i].cpudemand == 2))
			{
				tempplan.cpusum += temp[i].cpudemand;
				tempplan.memsum += temp[i].memdemand;
				tempplan.placeplan.push_back(temp[i]);
				temp.erase(temp.begin() + i);
				i = 0;
				bili = (ps.memcap - tempplan.memsum) / (ps.cpucap - tempplan.cpusum);
			}
		}
		else if (abs(bili - 4) <= abs(bili - 1) && abs(bili - 4) <= abs(bili - 2))
		{
			if ((tempplan.cpusum + temp[i].cpudemand <= ps.cpucap) &&
				(tempplan.memsum + temp[i].memdemand <= ps.memcap) &&
				(temp[i].memdemand / temp[i].cpudemand == 4))
			{
				tempplan.cpusum += temp[i].cpudemand;
				tempplan.memsum += temp[i].memdemand;
				tempplan.placeplan.push_back(temp[i]);
				temp.erase(temp.begin() + i);
				i = 0;
				bili = (ps.memcap - tempplan.memsum) / (ps.cpucap - tempplan.cpusum);
			}
		}


		/*if (i == temp.size()-1&&
		(tempplan.cpusum + needed_palce[i].cpudemand <= ps.cpucap) &&
		(tempplan.memsum + needed_palce[i].memdemand <= ps.memcap))
		{
		tempplan.cpusum += needed_palce[i].cpudemand;
		tempplan.memsum += needed_palce[i].memdemand;
		tempplan.placeplan.push_back(needed_palce[i]);
		}*/
	}
	return tempplan;
}

vector<Flavor>remove_max(vector<Flavor>need_palce, vector<Flavor>currentmax)
{
	int j = 0;
	while (currentmax.size() != 0)
	{
		if (currentmax[0].id == need_palce[j].id)
		{
			currentmax.erase(currentmax.begin());
			need_palce.erase(need_palce.begin() + j);
			j = 0;
		}
		else
			j++;
	}
	return need_palce;
}

int searchps(vector<Server>pstemp, vector<Flavor>need_place)
{
	double cpusumnow = 0, memsumnow = 0, bilinow;
	for (int i = 0; i < need_place.size(); i++)
	{
		cpusumnow += need_place[i].cpudemand;
		memsumnow += need_place[i].memdemand;
	}
	bilinow = memsumnow / cpusumnow;
	double min = abs(bilinow - pstemp[0].memcap / pstemp[0].cpucap);
	int minindex = 0;
	for (int i = 1; i < pstemp.size(); i++)
	{
		if (min > abs(bilinow - pstemp[i].memcap / pstemp[i].cpucap))
		{
			min = abs(bilinow - pstemp[i].memcap / pstemp[i].cpucap);
			minindex = i;
		}
	}
	return minindex;
}

void place(vector<Server> ps, vector<Flavor> fla_predict, char *filename)
{
	string amountperflavorstr;
	vector<Flavor>fla_needed_palce;
	vector<Flavor>temp_flavor = fla_predict;

	for (int i = temp_flavor.size() - 1; i >= 0; i--)
	{
		while (temp_flavor[i].num > 0)
		{
			fla_needed_palce.push_back(temp_flavor[i]);
			temp_flavor[i].num = temp_flavor[i].num - 1;
		}
	}

	vector<Server>Physicalserver = ps;
	vector<Flavor>temp_fla_needed_palce = fla_needed_palce;


	


	while (temp_fla_needed_palce.size() >0)
	{
		vector<plan>bestplan;
		for (int i = 0; i < Physicalserver.size(); i++)
		{
			plan besttemp = search_max(ps[i], temp_fla_needed_palce);
			bestplan.push_back(besttemp);
		}
		int bestplan_index = bestplan.size()-1;
		for (int j = 0; j < bestplan.size()-1; j++)
		{
			if (((double)(bestplan[j].cpusum*bestplan[j].memsum) / (double)(Physicalserver[j].cpucap*Physicalserver[j].memcap))
		>((double)(bestplan[bestplan_index].cpusum*bestplan[bestplan_index].memsum) /
		(double)(Physicalserver[bestplan_index].cpucap*Physicalserver[bestplan_index].memcap)))
			{
				bestplan_index = j;
			}
		}

		//	double max1 = searchmax.fun(0, maxplan);
		Physicalserver[bestplan_index].isuse = true;
		vector<Flavor> current_max = bestplan[bestplan_index].placeplan;
		Physicalserver[bestplan_index].num++;
		cout << Physicalserver[bestplan_index].name << ": CPU利用率"
			<< (double)((double)bestplan[bestplan_index].cpusum / (double)Physicalserver[bestplan_index].cpucap) <<
			"  MEM利用率  " << (double)((double)bestplan[bestplan_index].memsum / (double)Physicalserver[bestplan_index].memcap) << endl;
		Physicalserver[bestplan_index].distplan.push_back(bestplan[bestplan_index]);
		
		if (temp_fla_needed_palce.size() == current_max.size())
		{
			if (current_max.size() <= 3 && ((double)bestplan[bestplan_index].cpusum / (double)Physicalserver[bestplan_index].cpucap)
				+ (double)((double)bestplan[bestplan_index].memsum / (double)Physicalserver[bestplan_index].memcap)<0.8)
			{
				sum_vm -= current_max.size();
				for (int i = 0; i < current_max.size(); i++)
				{
					fla_predict[current_max[i].id - 1].num--;
				}
				Physicalserver[bestplan_index].distplan.erase(Physicalserver[bestplan_index].distplan.end() - 1);
				Physicalserver[bestplan_index].num--;
				if (Physicalserver[bestplan_index].num == 0)
				{
					Physicalserver[bestplan_index].isuse = false;
				}
			}
		}
		temp_fla_needed_palce = remove_max(temp_fla_needed_palce, current_max);
	}

	while (temp_fla_needed_palce.size() > 0)
	{
		int nowbest = searchps(Physicalserver, temp_fla_needed_palce);
		plan besttemp = search_max(Physicalserver[nowbest], temp_fla_needed_palce);
		Physicalserver[nowbest].isuse = true;
		vector<Flavor> current_max = besttemp.placeplan;

		Physicalserver[nowbest].num++;
		cout << Physicalserver[nowbest].name << ": CPU利用率"
			<< (double)((double)besttemp.cpusum / (double)Physicalserver[nowbest].cpucap) <<
			"  MEM利用率  " << (double)((double)besttemp.memsum / (double)Physicalserver[nowbest].memcap) << endl;
		Physicalserver[nowbest].distplan.push_back(besttemp);

		if (temp_fla_needed_palce.size() == current_max.size())
		{
			if (current_max.size() <= 3 && ((double)besttemp.cpusum / (double)Physicalserver[nowbest].cpucap)
				+ (double)((double)besttemp.memsum / (double)Physicalserver[nowbest].memcap)<0.5)
			{
				sum_vm -= current_max.size();
				for (int i = 0; i < current_max.size(); i++)
				{
					fla_predict[current_max[i].id - 1].num--;
				}
				Physicalserver[nowbest].distplan.erase(Physicalserver[nowbest].distplan.end() - 1);
				Physicalserver[nowbest].num--;
				if (Physicalserver[nowbest].num == 0)
				{
					Physicalserver[nowbest].isuse = false;
				}
			}
		}

		temp_fla_needed_palce = remove_max(temp_fla_needed_palce, current_max);

	}


	vector<int> a(19, 0);
	for (int i = 0; i < fla_predict.size(); ++i)
	{
		// printf("fla_predict[%d].num: %d, a[%d]: %d\n", i, fla_predict[i].num, i+1, a[i+1]);
		fla_predict[i].num -= a[fla_predict[i].id];
		cout << fla_predict[i].num << endl;
		amountperflavorstr += fla_predict[i].name + " " +
			to_string(fla_predict[i].num) + "\n";
	}
	string flavordistribution;
	for (size_t i = 0; i < Physicalserver.size(); ++i) {
		if (Physicalserver[i].isuse) {
			flavordistribution += Physicalserver[i].name + ' ' +
				to_string(Physicalserver[i].num) + "\n";
			for (int j = 1; j <= Physicalserver[i].num; ++j) {
				map<string, int> flavoramountindistribution;
				for (auto beg = Physicalserver[i].distplan[j-1].placeplan.cbegin();
					 beg != Physicalserver[i].distplan[j-1].placeplan.cend();
					 ++beg) {
					flavoramountindistribution[beg->name]++;
				}
				flavordistribution += Physicalserver[i].name + '-' +
					to_string(j) + ' ';
				for (auto beg = flavoramountindistribution.cbegin();
					 beg != flavoramountindistribution.cend();
					 ++beg)
					flavordistribution += beg->first + " " +
						to_string(beg->second) + " ";
				flavordistribution += "\n";
			}
			flavordistribution += "\n";
		}
	}

	cout << "sum_vm: " << sum_vm << endl;
	string result = to_string(sum_vm) + "\n" + amountperflavorstr + "\n" +
					flavordistribution;
	cout << "filename: " << filename << endl;
	write_result(result.c_str(), filename);
}

void predict_server(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num, char * filename) {
	DataFormat frmt_data = processdata(data, data_num);
	vector<vector<int>> historydata = get<0>(frmt_data);
	string starthistorydate = get<1>(frmt_data);
	string endhistorydate = get<2>(frmt_data);
#ifdef _DEBUG
	print_array(historydata);
	cout << "start history date: " << starthistorydate << "\nend history date: "
		<< endhistorydate << endl;
#endif

	InfoFormat frmt_info = processinfo(info);
	map<string, vector<int>> resourceofservers = get<0>(frmt_info);
	vector<Server>ps;
	for (auto beg = resourceofservers.cbegin(); beg != resourceofservers.cend(); ++beg) {
#ifdef _DEBUG
		cout << beg->first << " ";
		print_array(beg->second);
#endif
		Server ps_temp;
		ps_temp.name = beg->first;
		ps_temp.cpucap = beg->second[0];
		ps_temp.memcap = beg->second[1];
		ps_temp.harddiskcap = beg->second[2];
		ps.push_back(ps_temp);
	}
	int days = get<1>(frmt_info);
	vector<Flavor> flavorvec = get<2>(frmt_info);
	vector<int> flavoridvec;
	for (size_t i = 0; i < flavorvec.size(); ++i) {
		flavoridvec.push_back(flavorvec[i].id);
	}
	string startpredictdate = get<3>(frmt_info);
	int hourofstartdate = get<4>(frmt_info);
	int daydifference = day_relative_to_initialdate(startpredictdate,
		starthistorydate) -
		day_relative_to_initialdate(endhistorydate,
		starthistorydate);
	if (hourofstartdate > 22) {
		daydifference++;
	}

#ifdef _DEBUG
	print_array(flavoridvec);
	cout << "days: " << days << endl;
	cout << "start predict date: " << startpredictdate << endl;
	cout << "end day to start predict day difference: " << daydifference << endl;
#endif

	//hz

	 vector<Flavor> flavor_predict;

	 for (int i = 0; i < flavoridvec.size(); i++)
	 {
		 Flavor temp = flavorvec[i];
		 
		 //指数平滑预测
		/* vector<double>innow = predict_history(historydata, flavoridvec[i] - 1);
		 temp.num = predictzhishu(innow, days);*/
		 
		 //AR预测
		 temp.num = predict_AR(historydata, days, flavoridvec[i] - 1, AR_DAYS, daydifference);
		
		 //LR预测
		 temp.num = predict_LR(historydata, days, flavoridvec[i] - 1, LR_DAYS, daydifference);

		 //平均数预测
		 //temp.num = predict_sum(historydata, days, flavoridvec[i] - 1, average, daydifference)+5;

		 temp.memdemand = temp.memdemand / 1024;
		 sum_vm += temp.num;
		 flavor_predict.push_back(temp);
	 }

	 
	 place(ps, flavor_predict, filename);
}

DataFormat processdata(char *data[MAX_DATA_NUM], int data_num) {
	// Note: It is ambigous to be thought of as a declaration of function.
	// Refer to https://stackoverflow.com/a/9611296/6099429
	// istringstream record(string(data[0]));
	istringstream record(data[0]);
	string id, flavor, date, timeinday;
	record >> id >> flavor >> date >> timeinday;
	string initialdate = date;
	// Suppose the data record is in time order.
	int initialday = day_relative_to_initialdate(initialdate, initialdate);
	int lastday = initialday;
	vector<vector<int>> historydata;
	historydata.push_back(vector<int>(TOTAL_FLAVOR_TYPE_NUM));
	for (int i = 0; i < data_num; ++i) {
		istringstream record(data[i]);
		record >> id >> flavor >> date >> timeinday;
		int currday = day_relative_to_initialdate(date, initialdate);
		// If there is a date skipping
		for (int i = 0; i < currday - lastday; i++) {
			historydata.push_back(vector<int>(TOTAL_FLAVOR_TYPE_NUM));
		}
		lastday = currday;
		if (stoi(flavor.substr(6)) > TOTAL_FLAVOR_TYPE_NUM)
			continue;
		// Note: The reuslt of accessing an out-of-range subscript of vector is
		// undefined. It may works on Linux environment, but it may fail on
		// Windows. So taking care when accessing index of vector.
		historydata[currday - initialday][stoi(flavor.substr(6)) - 1]++;
	}
	// `initialdate` is the start date of history data, and `date` is the last
	// date of the history data
	return DataFormat(historydata, initialdate, date);
}

InfoFormat processinfo(char *info[MAX_INFO_NUM]) {
	int lnofstn = 0;      // line number of server type number
	istringstream record(info[lnofstn]);
	int typenum;
	record >> typenum;
	map<string, vector<int>> resourceofservers;
	for (int i = 0; i < typenum; i++) {
		record.clear(); record.str(info[lnofstn + i + 1]);
		string name;
		int kernelamount, memsize, diskvolume;
		record >> name >> kernelamount >> memsize >> diskvolume;
		vector<int> resourceperserver = { kernelamount, memsize, diskvolume };
		resourceofservers[name] = resourceperserver;
	}
	int lnoffa = lnofstn + typenum + 2; // line number of flavor amount
	record.clear(); record.str(info[lnoffa]);
	int flavoramount;
	record >> flavoramount;
	vector<Flavor> flavorvec;
	Flavor flvr;
	for (int i = 1; i <= flavoramount; i++) {
		istringstream record(info[lnoffa + i]);
		record >> flvr.name >> flvr.cpudemand >> flvr.memdemand;
		flvr.id = stoi(flvr.name.substr(6));
		flavorvec.push_back(flvr);
	}
	int lnoftime = lnoffa + flavoramount + 2;   // line number of prediction time
	record.clear(); record.str(info[lnoftime]);
	string datebegin, dateend;
	int hourofdatebegin, hourofdateend;
	record >> datebegin >> hourofdatebegin;
	record.clear(); record.str(info[lnoftime + 1]);
	record >> dateend >> hourofdateend;
#ifdef _DEBUG
	cout << "hour of date begin: " << hourofdatebegin << "\nhour of date end: "
		<< hourofdateend << endl;
#endif
	int days = day_of_year(dateend) - day_of_year(datebegin);
	// Note the time of prediction date format will be 00:00:00 or 23:59:59, so
	// needing to check it by a close difference 22.
	// Refer to https://forum.huaweicloud.com/thread-8311-1-1.html
	if (hourofdateend - hourofdatebegin > 22)
		days++;
	else if (hourofdateend - hourofdatebegin < -22) {
		days--;
	}
	return InfoFormat(resourceofservers, days, flavorvec, datebegin, hourofdatebegin);
}

/* vim: set noet sts=4 sw=4 ts=4 tw=80 fileformat=dos: */

//求矩阵的转置
double **Transpose(double **matrix_, int M, int N)
{
	double **matrix_t = new double*[N];
	for (int m = 0; m < N; m++)
	{
		matrix_t[m] = new double[M];
	}
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			matrix_t[j][i] = matrix_[i][j];
		}
	}
	return matrix_t;
}

//求矩阵点乘
double **Multiply(double**matrix_x, double**matrix_y, int X, int Y, int Z)
{
	//X*Y的矩阵和Y*Z的矩阵点乘；
	double **matrix_result = new double*[X];
	for (int i = 0; i < X; i++)
	{
		matrix_result[i] = new double[Z];
	}//输出矩阵为X*Z
	for (int m = 0; m<X; m++)
	{
		for (int s = 0; s<Z; s++)
		{
			matrix_result[m][s] = 0;//变量使用前记得初始化,否则结果具有不确定性  
			for (int n = 0; n<Y; n++){
				matrix_result[m][s] += matrix_x[m][n] * matrix_y[n][s];
			}
		}
	}
	return matrix_result;
}

//LUP分解
void LUP_Descomposition(double **A_temp, double **L, double **U, int *P, int N)
{
	int row = 0;
	for (int i = 0; i<N; i++)
	{
		P[i] = i;
	}

	for (int i = 0; i<N - 1; i++)
	{
		double p = 0.0;
		for (int j = i; j<N; j++)
		{
			if (abs(A_temp[j][i])>p)
			{
				p = abs(A_temp[j][i]);
				row = j;
			}
		}
		if (0 == p)
		{
			cout << "矩阵奇异，无法计算逆" << endl;
			return;
		}
		//交换P[i]和P[row]
		int tmp = P[i];
		P[i] = P[row];
		P[row] = tmp;

		double tmp2 = 0.0;
		for (int j = 0; j<N; j++)
		{
			//交换A[i][j]和 A[row][j]
			tmp2 = A_temp[i][j];
			A_temp[i][j] = A_temp[row][j];
			A_temp[row][j] = tmp2;
		}

		//以下同LU分解
		double u = A_temp[i][i], l = 0.0;
		for (int j = i + 1; j<N; j++)
		{
			l = A_temp[j][i] / u;
			A_temp[j][i] = l;
			for (int k = i + 1; k<N; k++)
			{
				A_temp[j][k] = A_temp[j][k] - A_temp[i][k] * l;
			}
		}

	}

	//构造L和U
	for (int i = 0; i<N; i++)
	{
		for (int j = 0; j <= i; j++)
		{
			if (i != j)
			{
				L[i][j] = A_temp[i][j];
			}
			else
			{
				L[i][j] = 1;
			}
		}
		for (int k = i; k<N; k++)
		{
			U[i][k] = A_temp[i][k];
		}
	}

}

//LUP求解方程
double * LUP_Solve(double **L, double **U, int *P, double *b, int N)
{
	double *x = new double[N];
	double *y = new double[N];

	//正向替换
	for (int i = 0; i < N; i++)
	{
		y[i] = b[P[i]];
		for (int j = 0; j < i; j++)
		{
			y[i] = y[i] - L[i][j] * y[j];
		}
	}
	//反向替换
	for (int i = N - 1; i >= 0; i--)
	{
		x[i] = y[i];
		for (int j = N - 1; j > i; j--)
		{
			x[i] = x[i] - U[i][j] * x[j];
		}
		x[i] /= U[i][i];
	}
	return x;
}
//矩阵求逆
double ** inverse(double** A, int N)
{
	//创建矩阵A的副本，注意不能直接用A计算，因为LUP分解算法已将其改变
	double **A_mirror = new double*[N];
	double **inv_A = new double*[N];//最终的逆矩阵（还需要转置）
	for (int i = 0; i < N; i++)
	{
		inv_A[i] = new double[N];
		A_mirror[i] = new double[N];
	}
	double *inv_A_each = new double[N];//矩阵逆的各列
	double *b = new double[N];//b阵为B阵的列矩阵分量
	for (int i = 0; i<N; i++)
	{
		double **L = new double*[N];
		double **U = new double*[N];
		for (int m = 0; m < N; m++)
		{
			L[m] = new double[N];
			U[m] = new double[N];
		}
		int *P = new int[N];


		//构造单位阵的每一列
		for (int w = 0; w<N; w++)
		{
			b[w] = 0;
		}
		b[i] = 1;

		//每次都需要重新将A复制一份
		for (int i = 0; i<N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				A_mirror[i][j] = A[i][j];
			}
		}

		LUP_Descomposition(A_mirror, L, U, P, N);

		inv_A_each = LUP_Solve(L, U, P, b, N);

		memcpy(inv_A[i], inv_A_each, N*sizeof(double));//将各列拼接起来
	}
	inv_A = Transpose(inv_A, N, N);//由于现在根据每列b算出的x按行存储，因此需转置

	return inv_A;
}

//求ar模型的参数
vector<double> calculateA(vector<double>in, int p)
{
	int n = in.size();
	vector<double>A;
	//x为（n-p)*p的矩阵，y为(n-p)*1的矩阵，a为p*1的矩阵
	double **matrix_x = new double*[n - p];
	double **matrix_y = new double*[n - p];
	double **a;
	for (int i = 0; i < n - p; i++)
	{
		matrix_x[i] = new double[p];
		matrix_y[i] = new double[1];
	}
	for (int i = 0; i < n - p; i++)
	{
		for (int j = 0; j < p; j++)
		{
			matrix_x[i][j] = in[j + i];//x矩阵
		}
		matrix_y[i][0] = in[p + i];
	}
	double**matrix_xt = Transpose(matrix_x, n - p, p);
	double **ata = Multiply(matrix_xt, matrix_x, p, n - p, p);
	a = Multiply(Multiply(inverse(ata, p), matrix_xt, p, p, n - p), matrix_y, p, n - p, 1);
	bool is = true;
	for (int i = 0; i < p; i++)
	{
		if (abs(a[i][0])>10)
		{
			is = false;
		}
	}
	if (is)
	{
		for (int i = 0; i < p; i++)
		{
			A.push_back(a[i][0]);
		}
	}
	else if (!is)
	{
		for (int i = 0; i < p; i++)
		{
			double te = 1.0 / p;
			A.push_back(te);
		}
	}

	return A;
}

double piancha(vector<double>in, int p, vector<double>A)
{
	int n = in.size();
	double error = 0.0;
	for (int i = 0; i < n - p; i++)
	{
		double yuce = 0.0;
		for (int j = 0; j < p; j++)
		{
			yuce += A[j] * in[i + j];
		}
		error += in[i + p] - yuce;
	}

	error = error / (n - p);
	return error;
}