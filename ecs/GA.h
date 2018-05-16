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
#include<ctime>
#define GROUP_NUM 100				// 群体规模

const double P_INHERIATANCE = 0.200;	// 变异概率
const double P_COPULATION = 0.900004;	// 杂交概率
const int ITERATION_NUM = 50;		// 迭代次数
const double MAX_INT = 9999999.0;
int IndexCross_i;
int IndexCross_j;//交叉操作基因片段

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

typedef struct{
	double cost;//cpu利用率和mem利用率的和
	double fitness;//适应度值
	vector<int>path;//装箱顺序
	double Probability;//个体被选中的概率
}solution;


class GA
{
public:
	GA(vector<Server>currentps, vector<Flavor>currentflavor_needplace);
	~GA();
	vector<solution>group;//种群
	solution globalbest;//存储全局最优
	vector<solution> Son_solution;
	vector<Server>ps;
	vector<Flavor>flavor_needplace;
	int n;//染色体长度
	double cpudemandsum;
	double memdemandsum;
	plan search_max(Server ps, vector<Flavor>needed_palce);
	double CalculateCost(vector<Server> current_PS, vector<Flavor> need_place, solution current_solution);
	vector<Flavor>remove_max(vector<Flavor>need_palce, vector<Flavor>currentmax);
	vector<solution> Calc_Probablity(vector<solution>current_group);


	void InitialGroup(vector<Server> current_PS,vector<Flavor> need_place);//初始化
	int  Evo_Select();//交叉选择
	void Evo_Cross(solution Father, solution Mother);//交叉
	void Evo_Variation(int Index_Variation);		// 变异
	void Evolution(vector<Server> current_PS, vector<Flavor> need_place);
	void Evo_UpdateGroup(vector<Server> currentPS, vector<Flavor>need_place);//更新种群
	void Evaluate(vector<Server> currentPS, vector<Flavor>need_place);	//评价
	vector<Server>Calculatebestsolution();
};

GA::GA(vector<Server>currentps, vector<Flavor>currentflavor_needplace)
{
	cpudemandsum = 0;
	memdemandsum = 0;
	ps = currentps;
	flavor_needplace = currentflavor_needplace;
	n = currentflavor_needplace.size();
	for (int i = 0; i < n; i++)
	{
		cpudemandsum += currentflavor_needplace[i].cpudemand;
		memdemandsum += currentflavor_needplace[i].memdemand;
	}
}
GA::~GA()
{}

vector<Server>GA::Calculatebestsolution()
{
	double cpuuse = 0.0, memuse = 0.0;
	vector<Flavor>need_place0, need_place1, need_place2;
	vector<vector<Flavor>>flavor_part;
	int costofps = 1;
	int cpu_num = 0, mem_num = 0;
	for (int i = 0; i < globalbest.path.size(); i++)
	{
		if (globalbest.path[i] == 0)
		{
			need_place0.push_back(flavor_needplace[i]);
		}
		else if (globalbest.path[i] == 1)
		{
			need_place1.push_back(flavor_needplace[i]);
		}
		else if (globalbest.path[i] == 2)
		{
			need_place2.push_back(flavor_needplace[i]);
		}
	}
	flavor_part.push_back(need_place0);
	flavor_part.push_back(need_place1);
	flavor_part.push_back(need_place2);
	int i = 0;
	vector<Server>ps_temp;
	while (i <= 2)
	{
		while (flavor_part[i].size() > 0)
		{
			plan besttemp = search_max(ps[i], flavor_part[i]);
			ps[i].isuse = true;
			ps[i].num++;
			ps[i].distplan.push_back(besttemp);
			cpuuse += ps[i].cpucap;
			memuse += ps[i].memcap;
			flavor_part[i] = remove_max(flavor_part[i], besttemp.placeplan);
		}
		i++;
	}

	for (int i = 0; i < ps.size()-1; i++)
	{
		for (int j = ps.size()-1; j > i; j--)
		{
			for (int m = 0; m < ps.size(); m++)
			{
				if (ps[i].distplan[ps[i].distplan.size() - 1].cpusum
					+ ps[j].distplan[ps[j].distplan.size() - 1].cpusum <= ps[m].cpucap&&
					ps[i].distplan[ps[i].distplan.size() - 1].memsum
					+ ps[j].distplan[ps[j].distplan.size() - 1].memsum <= ps[m].memcap)
				{
					ps[i].num--;
					ps[j].num--;
					ps[m].num++;
					plan temp1 = ps[i].distplan[ps[i].distplan.size() - 1];
					plan temp2 = ps[j].distplan[ps[j].distplan.size() - 1];
					temp1.cpusum += temp2.cpusum;
					temp1.memsum += temp2.memsum;
					for (int q = 0; q < temp2.placeplan.size(); q++)
					{
						temp1.placeplan.push_back(temp2.placeplan[q]);
					}
					ps[i].distplan.erase(ps[i].distplan.end() - 1);
					ps[j].distplan.erase(ps[j].distplan.end() - 1);
					ps[m].distplan.push_back(temp1);
				}
			}
			
		}

	}
	return ps;
}

/*double abs(double a)
{
	if (a < 0)
	{
		return (-a);
	}
	else
		return a;
}*/
plan GA::search_max(Server ps, vector<Flavor>needed_palce)
{
	plan tempplan;
	vector<Flavor>temp = needed_palce;
	tempplan.cpusum += temp[0].cpudemand;
	tempplan.memsum += temp[0].memdemand;
	tempplan.placeplan.push_back(temp[0]);
	temp.erase(temp.begin());
	if (temp.size() == 0)
	{
		return tempplan;
	}
	double bili = (ps.memcap - tempplan.memsum) / (ps.cpucap - tempplan.cpusum);
	for (int i = 0; i < temp.size()+1; i++)
	{
		if (i == temp.size())
		{
			if ((tempplan.cpusum + temp[i-1].cpudemand <= ps.cpucap) &&
				(tempplan.memsum + temp[i-1].memdemand <= ps.memcap))
			{
				tempplan.cpusum += temp[i-1].cpudemand;
				tempplan.memsum += temp[i-1].memdemand;
				tempplan.placeplan.push_back(temp[i-1]);
				temp.erase(temp.end()-1);
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


		if (abs(bili - 1) <= abs(bili - 2) && abs(bili - 1)<=abs(bili - 4))
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
				(temp[i].memdemand / temp[i].cpudemand ==2 ))
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

/*plan GA::search_max(Server ps, vector<Flavor>needed_palce)
{
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
				((pack[j - needed_palce[i].cpudemand].cpusum + needed_palce[i].cpudemand)/ps.cpucap+
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
vector<Flavor>GA::remove_max(vector<Flavor>need_palce, vector<Flavor>currentmax)
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

double GA::CalculateCost(vector<Server> current_PS, vector<Flavor> need_place, solution current_solution)
{
	double cpuuse = 0.0, memuse = 0.0;
	vector<Flavor>need_place0, need_place1, need_place2;
	vector<vector<Flavor>>flavor_part;
	int costofps = 1;
	int cpu_num = 0, mem_num = 0;
	for (int i = 0; i < current_solution.path.size(); i++)
	{
		if (current_solution.path[i] == 0)
		{
			need_place0.push_back(need_place[i]);
		}
		else if (current_solution.path[i] == 1)
		{
			need_place1.push_back(need_place[i]);
		}
		else if (current_solution.path[i] == 2)
		{
			need_place2.push_back(need_place[i]);
		}
	}
	flavor_part.push_back(need_place0);
	flavor_part.push_back(need_place1);
	flavor_part.push_back(need_place2);
	int i = 0;
	vector<Server>ps_temp;
	while (i <= 2)
	{
		while (flavor_part[i].size() > 0)
		{
			plan besttemp = search_max(current_PS[i], flavor_part[i]);
			cpuuse += current_PS[i].cpucap;
			memuse += current_PS[i].memcap;
			flavor_part[i] = remove_max(flavor_part[i], besttemp.placeplan);
		}
		i++;
	}

	return (cpudemandsum / cpuuse + memdemandsum / memuse);
}

void GA::InitialGroup(vector<Server> current_PS, vector<Flavor> need_place)
{
	for (int i = 0; i < GROUP_NUM; i++)
	{
		solution temp_solution;
		srand(time(NULL));
		for (int j = 0; j < flavor_needplace.size(); j++)
		{
			if (flavor_needplace[j].id == 18)
			{
				temp_solution.path.push_back(2);
			}
			else if (flavor_needplace[j].memdemand / flavor_needplace[j].cpudemand == 1)
			{
				temp_solution.path.push_back((rand() % (current_PS.size() - 1)));
			}
			else if (flavor_needplace[j].memdemand / flavor_needplace[j].cpudemand == 2)
			{
				temp_solution.path.push_back((rand() % (current_PS.size())));
			}
			else if (flavor_needplace[j].memdemand / flavor_needplace[j].cpudemand == 4)
			{
				int temp = 2*(rand()%2);
				temp_solution.path.push_back(temp);
			}
		}
		temp_solution.cost = CalculateCost(current_PS, need_place, temp_solution);	//计算一个方案的费用

		group.push_back(temp_solution);
	}

	group = Calc_Probablity(group);
	globalbest = group[0];//给全局最优赋初值
}

vector<solution> GA::Calc_Probablity(vector<solution>current_group)
{
	double max_cost = 0;
	double totalcost = 0;

	for (int i = 0; i < current_group.size(); i++)
	{
		max_cost = max_cost > current_group[i].cost ? max_cost : current_group[i].cost;//找到最大
		totalcost += current_group[i].cost;//计算成本之和
	}
	for (int i = 0; i < current_group.size(); i++)
	{
		current_group[i].Probability = (double)((current_group[i].cost))
			/ (totalcost);
	}
	return current_group;
}

int  GA::Evo_Select()
{
	double selection_P = ((rand() % 100 + 0.0) / 100);
	double distribution_P = 0.0;
	for (int i = 0; i < group.size(); i++)
	{
		distribution_P += group[i].Probability;
		if (selection_P < distribution_P)
		{
			return i;
		}
	}
	cout << "【ERROR!】Evo_Select() 轮盘赌选择有误..." << endl;
	return 0;
}

void GA::Evolution(vector<Server> current_PS, vector<Flavor> need_place)
{
	int iter = 0;
	while (iter < ITERATION_NUM)
	{
		cout << "***********************【第次" << (iter + 1) << "迭代】*************************" << endl;
		// 1. 选择
		int Father_index = Evo_Select();
		int Mother_index = Evo_Select();

		while (Mother_index == Father_index)
		{
			// 防止Father和Mother都是同一个个体 -> 自交(父母为同一个个体时, 母亲重新选择, 直到父母为不同的个体为止 )
			Mother_index = Evo_Select();
		}

		// groups[]为当前种群
		solution Father = group[Father_index];
		solution Mother = group[Mother_index];

		// 2. 交叉, 存储在全局变量 Son_solution[] 数组 - 通过M次杂交, 产生2M个新个体, 2M >= GROUP_NUM
		int M = GROUP_NUM / 2; 
		vector<solution>newsun;
		Son_solution = newsun;
		while (M >= 0)
		{
			double Is_COPULATION = ((rand() % 100 + 0.0) / 100);
			if (Is_COPULATION < P_COPULATION)//以一定概率判断是否进行杂交
				// 杂交, 将结果存储于遗传个体总群,全局变量Son_solution[]
				Evo_Cross(Father, Mother);
			else
			{
				Son_solution.push_back(Father);
				Son_solution.push_back(Mother);
			}
			M--;

		}

		// 3. 变异：针对 Son_solution[]
		for (int IndexVariation = 0; IndexVariation < Son_solution.size(); IndexVariation++)
		{
			double RateVariation = double((rand() % 100)) / 100;
			// 产生的随机数小于变异概率 则该个体进行变异
			if (RateVariation < P_INHERIATANCE)
			{
				Evo_Variation(IndexVariation);
				// 产生新个体, 计算新的费用
				Son_solution[IndexVariation].cost = CalculateCost(current_PS, need_place, Son_solution[IndexVariation]);	//计算一条路线的费用
			}
		}
		// 4. 更新群体
		// 参与对象：父代 + 遗传的子代
		Evo_UpdateGroup(current_PS, need_place);
		iter++;
	}
}

void GA::Evo_Cross(solution Father, solution Mother)
{

	// 杂交过程：随机产生杂交的位置, 保证 IndexCross_i < IndexCross_j
	IndexCross_i = rand() % (n);
	IndexCross_j = rand() % (n);
	if (IndexCross_i > IndexCross_j)
	{
		int temp = IndexCross_i;
		IndexCross_i = IndexCross_j;
		IndexCross_j = temp;
	}
	if (IndexCross_j == n)
	{
		cout << "[ 杂交过程的随机数产生有问题... ]" << endl;
	}

	// 杂交基因段
	int *Father_Cross = new int[IndexCross_j - IndexCross_i + 1];	// 父亲交叉基因段
	int *Mother_Cross = new int[IndexCross_j - IndexCross_i + 1];// 母亲交叉基因段
	int Length_Cross = 0;		// 杂交片段基因的个数（j-i)
	for (int i = IndexCross_i; i <= IndexCross_j; i++)
	{
		Father_Cross[Length_Cross] = Father.path[i];
		Mother_Cross[Length_Cross] = Mother.path[i];
		Length_Cross++;
	}

	// Father and Mother 交换基因段
	int _temp;
	solution Father_temp = Father;
	solution Mother_temp = Mother;
	for (int i = IndexCross_i; i <= IndexCross_j; i++)
	{
		_temp = Father_temp.path[i];
		Father_temp.path[i] = Mother_temp.path[i];
		Mother_temp.path[i] = _temp;
	}

	// 开始杂交 - 处理 Mother, 其中Length_Conflict会在函数Get_Conflict()中改变并保存
	
	Son_solution.push_back(Father_temp);
	Son_solution.push_back(Mother_temp);
}

void GA::Evo_Variation(int Index_Variation){
	// 随机产生两个随机数表示两个基因的位置, 并进行位置交换
	int Variation_i = (rand() % (n));
	int Variation_j = (rand() % (n));

	while (Variation_i == Variation_j){
		Variation_j = (rand() % (n));
	}
	//变异
	int temp = Son_solution[Index_Variation].path[Variation_i];
	Son_solution[Index_Variation].path[Variation_i] = Son_solution[Index_Variation].path[Variation_j];
	Son_solution[Index_Variation].path[Variation_j] = temp;
}

void GA::Evo_UpdateGroup(vector<Server> currentPS, vector<Flavor>need_place)
{
	solution tempSolution;
	// 先对子代 - Son_solution[] 依据适应度进行排序 - 降序[按适应度从大到小]
	vector<solution>totalgroup = Son_solution;
	for (int m = 0; m < group.size(); m++)
	{
		totalgroup.push_back(group[m]);
	}
	for (int i = 0; i < totalgroup.size(); i++)
	{
		for (int j = 0; j<totalgroup.size() - i - 1; j++)
		{
			if (totalgroup[j].cost < totalgroup[j + 1].cost)
			{
				tempSolution = totalgroup[j+1];
				totalgroup[j+1] = totalgroup[j];
				totalgroup[j] = tempSolution;
			}
		}
	}
	// 更新
	// 适应度值越高，表现越好
	for (int i = 0; i < group.size(); i++)	// 父代 - 按适应度从大到小排序
	{
		group[i] = totalgroup[i];
	}
	//group = Calc_Probablity(group);//重新计算选择概率
	Evaluate(currentPS, need_place);
}


void GA::Evaluate(vector<Server> currentPS, vector<Flavor>need_place){
	solution bestSolution;
	bestSolution = group[0];
	for (int i = 1; i < group.size(); i++)
	{
		if (bestSolution.cost < group[i].cost)
		{
			bestSolution = group[i];
		}
	}
	if (globalbest.cost < bestSolution.cost)
	{
		globalbest = bestSolution;
	}

	cout << "当前全局最优解的利用率:" << globalbest.cost << endl;

	cout << endl;

}
