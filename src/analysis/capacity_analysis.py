"""
Implementation-Capacity-Equity Analysis
========================================

Research Question:
"Do countries have the capacity to implement their AI policies, 
and how does this vary between high-income and developing countries?"

This script extracts implementation capacity indicators from policy text
and creates country-level capacity scores.

Capacity Indicators (text-based):
1. INSTITUTIONAL: Dedicated agencies, coordination bodies
2. ENFORCEMENT: Penalties, audits, compliance mechanisms
3. RESOURCES: Budget mentions, funding, staff
4. OPERATIONAL: Timelines, milestones, monitoring
5. EXPERTISE: Technical standards, partnerships, research

Ambition Indicators:
1. SCOPE: Sectors covered, breadth of application
2. BINDING: Legal force (binding vs voluntary)
3. GOALS: Stated objectives and targets
"""

import json
import re
import logging
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Country income classifications (World Bank 2024)
HIGH_INCOME = {
    'United States', 'United Kingdom', 'Germany', 'France', 'Japan', 'Canada', 
    'Australia', 'Singapore', 'Netherlands', 'Sweden', 'Switzerland', 'Norway', 
    'Denmark', 'Finland', 'Austria', 'Belgium', 'Ireland', 'New Zealand', 'Korea',
    'Israel', 'Italy', 'Spain', 'Portugal', 'Czech Republic', 'Slovenia', 'Estonia',
    'Luxembourg', 'Malta', 'Cyprus', 'Iceland', 'Saudi Arabia', 'United Arab Emirates',
    'Chile', 'Uruguay', 'Croatia', 'Lithuania', 'Latvia', 'Slovak Republic', 'Poland',
    'Hungary', 'Greece',
    # Organizations (treat as HIC)
    'European Union', 'OECD/GPAI', 'G7'
}

UPPER_MIDDLE = {
    'China (People\'s Republic of)', 'Brazil', 'Mexico', 'Türkiye', 'Argentina',
    'Colombia', 'Thailand', 'Malaysia', 'Peru', 'Romania', 'Bulgaria', 'Serbia',
    'Costa Rica', 'Kazakhstan', 'South Africa', 'Mauritius'
}

LOWER_MIDDLE = {
    'India', 'Indonesia', 'Viet Nam', 'Ukraine', 'Egypt', 'Morocco', 'Tunisia',
    'Nigeria', 'Kenya', 'Philippines', 'Uzbekistan', 'Algeria', 'Armenia'
}

LOW_INCOME = {
    'Rwanda', 'Uganda', 'Zambia'
}

# Capacity indicator keywords/patterns
CAPACITY_INDICATORS = {
    'institutional': {
        'agency_dedicated': [
            r'ai\s+agency', r'ai\s+authority', r'ai\s+office', r'ai\s+unit',
            r'dedicated\s+(?:agency|body|unit|office)', r'regulatory\s+authority',
            r'data\s+protection\s+authority', r'supervisory\s+authority',
            r'national\s+ai\s+(?:center|centre|institute)', r'ai\s+commission'
        ],
        'coordination': [
            r'inter-?ministerial', r'cross-?ministry', r'coordination\s+(?:body|committee|council)',
            r'working\s+group', r'task\s+force', r'steering\s+committee',
            r'national\s+council', r'advisory\s+(?:board|council|committee)'
        ]
    },
    'enforcement': {
        'penalties': [
            r'fine[sd]?', r'penalt(?:y|ies)', r'sanction[s]?', r'enforcement\s+action',
            r'administrative\s+(?:fine|penalty)', r'monetary\s+(?:penalty|sanction)',
            r'infringement', r'non-?compliance', r'violation[s]?'
        ],
        'audit': [
            r'audit[s]?', r'inspection[s]?', r'compliance\s+(?:check|review|assessment)',
            r'impact\s+assessment', r'algorithmic\s+audit', r'third-?party\s+audit',
            r'conformity\s+assessment', r'certification'
        ],
        'mechanisms': [
            r'enforcement\s+mechanism', r'redress', r'complaint[s]?\s+(?:mechanism|procedure)',
            r'appeal[s]?', r'judicial\s+review', r'ombudsman', r'dispute\s+resolution'
        ]
    },
    'resources': {
        'budget': [
            r'budget', r'funding', r'allocation', r'investment',
            r'(?:million|billion|€|£|\$|EUR|USD)\s*\d+', r'\d+\s*(?:million|billion|€|£|\$)',
            r'financial\s+(?:resources|support)', r'appropriation'
        ],
        'staff': [
            r'staff(?:ing)?', r'personnel', r'employees?', r'workforce',
            r'human\s+resources', r'recruit(?:ment|ing)?', r'hiring',
            r'full-?time\s+equivalent', r'FTE'
        ]
    },
    'operational': {
        'timeline': [
            r'deadline', r'by\s+\d{4}', r'within\s+\d+\s+(?:months?|years?)',
            r'timeline', r'milestone[s]?', r'phase[sd]?\s+(?:in|out|implementation)',
            r'entry\s+into\s+force', r'effective\s+date', r'transitional\s+period'
        ],
        'monitoring': [
            r'monitor(?:ing)?', r'evaluat(?:e|ion)', r'review\s+(?:mechanism|period|cycle)',
            r'reporting\s+(?:requirement|obligation)', r'annual\s+report',
            r'progress\s+(?:report|review)', r'KPI', r'indicator[s]?', r'metric[s]?'
        ]
    },
    'expertise': {
        'technical': [
            r'technical\s+(?:expertise|standard|specification)', r'certification\s+(?:body|scheme)',
            r'standard[s]?\s+(?:body|organization|development)', r'ISO', r'IEEE',
            r'technical\s+committee', r'expert\s+(?:group|panel|committee)'
        ],
        'research': [
            r'research\s+(?:partnership|collaboration|institute|center|centre)',
            r'university\s+partnership', r'academic\s+(?:partner|collaboration)',
            r'R&D', r'research\s+and\s+development', r'innovation\s+(?:hub|lab|center)'
        ]
    }
}

# Ambition indicators
AMBITION_INDICATORS = {
    'binding_keywords': [
        r'mandatory', r'binding', r'shall\s+(?:be|comply|ensure|provide)',
        r'must\s+(?:be|comply|ensure|provide)', r'required\s+to',
        r'obligation', r'prohibited', r'ban(?:ned|s)?'
    ],
    'voluntary_keywords': [
        r'voluntary', r'guideline[s]?', r'recommendation[s]?', r'encourage[sd]?',
        r'may\s+(?:be|comply)', r'should\s+(?:be|consider)', r'best\s+practice[s]?',
        r'principle[s]?', r'framework'
    ]
}


@dataclass
class PolicyCapacity:
    """Capacity assessment for a single policy."""
    title: str
    jurisdiction: str
    year: Optional[int]
    income_group: str
    
    # Capacity scores (0-1 normalized by max possible)
    institutional_score: float = 0.0
    enforcement_score: float = 0.0
    resources_score: float = 0.0
    operational_score: float = 0.0
    expertise_score: float = 0.0
    
    # Overall capacity
    total_capacity_score: float = 0.0
    
    # Ambition scores
    binding_score: float = 0.0  # Higher = more binding
    sector_breadth: int = 0  # Number of sectors
    
    # Raw counts for transparency
    indicators_found: Dict = field(default_factory=dict)
    word_count: int = 0


@dataclass 
class JurisdictionCapacity:
    """Aggregated capacity for a jurisdiction."""
    jurisdiction: str
    income_group: str
    policy_count: int = 0
    
    # Average scores
    avg_institutional: float = 0.0
    avg_enforcement: float = 0.0
    avg_resources: float = 0.0
    avg_operational: float = 0.0
    avg_expertise: float = 0.0
    avg_total_capacity: float = 0.0
    
    # Ambition
    avg_binding: float = 0.0
    avg_sector_breadth: float = 0.0
    
    # Gap analysis
    capacity_ambition_gap: float = 0.0  # Negative = under-capacity


class CapacityAnalyzer:
    """Extracts and scores implementation capacity from policy text."""
    
    def __init__(self, enriched_file: str, corpus_file: str):
        logger.info(f"Loading enriched data from {enriched_file}")
        with open(enriched_file, 'r', encoding='utf-8') as f:
            self.enriched = json.load(f)
        
        logger.info(f"Loading corpus from {corpus_file}")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            self.corpus = json.load(f)
        
        # Build lookup by URL
        self.corpus_lookup = {e['url']: e for e in self.corpus['entries']}
        
        self.policy_results: List[PolicyCapacity] = []
        self.jurisdiction_results: Dict[str, JurisdictionCapacity] = {}
    
    def get_income_group(self, jurisdiction: str) -> str:
        """Classify jurisdiction by income group."""
        if jurisdiction in HIGH_INCOME:
            return 'High Income'
        elif jurisdiction in UPPER_MIDDLE:
            return 'Upper Middle'
        elif jurisdiction in LOWER_MIDDLE:
            return 'Lower Middle'
        elif jurisdiction in LOW_INCOME:
            return 'Low Income'
        else:
            return 'Unclassified'
    
    def count_pattern_matches(self, text: str, patterns: List[str]) -> int:
        """Count how many patterns match in text."""
        text_lower = text.lower()
        count = 0
        for pattern in patterns:
            if re.search(pattern, text_lower):
                count += 1
        return count
    
    def analyze_policy(self, policy: dict) -> PolicyCapacity:
        """Analyze a single policy for capacity indicators."""
        url = policy.get('url', '')
        corpus_entry = self.corpus_lookup.get(url, {})
        
        # Get full text (corpus content + overview)
        text = corpus_entry.get('content', '') or policy.get('initiative_overview', '') or policy.get('description', '')
        
        jurisdiction = policy.get('jurisdiction', 'Unknown')
        income_group = self.get_income_group(jurisdiction)
        
        result = PolicyCapacity(
            title=policy.get('title', ''),
            jurisdiction=jurisdiction,
            year=policy.get('start_year'),
            income_group=income_group,
            word_count=len(text.split())
        )
        
        # Score each capacity dimension
        indicators_found = {}
        
        # Institutional
        agency = self.count_pattern_matches(text, CAPACITY_INDICATORS['institutional']['agency_dedicated'])
        coord = self.count_pattern_matches(text, CAPACITY_INDICATORS['institutional']['coordination'])
        result.institutional_score = min((agency + coord) / 4, 1.0)  # Normalize
        indicators_found['institutional'] = {'agency': agency, 'coordination': coord}
        
        # Enforcement
        penalties = self.count_pattern_matches(text, CAPACITY_INDICATORS['enforcement']['penalties'])
        audit = self.count_pattern_matches(text, CAPACITY_INDICATORS['enforcement']['audit'])
        mechanisms = self.count_pattern_matches(text, CAPACITY_INDICATORS['enforcement']['mechanisms'])
        result.enforcement_score = min((penalties + audit + mechanisms) / 6, 1.0)
        indicators_found['enforcement'] = {'penalties': penalties, 'audit': audit, 'mechanisms': mechanisms}
        
        # Resources
        budget = self.count_pattern_matches(text, CAPACITY_INDICATORS['resources']['budget'])
        staff = self.count_pattern_matches(text, CAPACITY_INDICATORS['resources']['staff'])
        result.resources_score = min((budget + staff) / 4, 1.0)
        indicators_found['resources'] = {'budget': budget, 'staff': staff}
        
        # Operational
        timeline = self.count_pattern_matches(text, CAPACITY_INDICATORS['operational']['timeline'])
        monitoring = self.count_pattern_matches(text, CAPACITY_INDICATORS['operational']['monitoring'])
        result.operational_score = min((timeline + monitoring) / 4, 1.0)
        indicators_found['operational'] = {'timeline': timeline, 'monitoring': monitoring}
        
        # Expertise
        technical = self.count_pattern_matches(text, CAPACITY_INDICATORS['expertise']['technical'])
        research = self.count_pattern_matches(text, CAPACITY_INDICATORS['expertise']['research'])
        result.expertise_score = min((technical + research) / 4, 1.0)
        indicators_found['expertise'] = {'technical': technical, 'research': research}
        
        # Total capacity (weighted average)
        result.total_capacity_score = (
            result.institutional_score * 0.25 +
            result.enforcement_score * 0.25 +
            result.resources_score * 0.20 +
            result.operational_score * 0.15 +
            result.expertise_score * 0.15
        )
        
        # Ambition scoring
        binding = self.count_pattern_matches(text, AMBITION_INDICATORS['binding_keywords'])
        voluntary = self.count_pattern_matches(text, AMBITION_INDICATORS['voluntary_keywords'])
        if binding + voluntary > 0:
            result.binding_score = binding / (binding + voluntary)
        
        result.sector_breadth = len(policy.get('target_sectors', []))
        
        result.indicators_found = indicators_found
        
        return result
    
    def analyze_all(self):
        """Analyze all policies."""
        logger.info(f"Analyzing {len(self.enriched['policies'])} policies...")
        
        for policy in self.enriched['policies']:
            result = self.analyze_policy(policy)
            self.policy_results.append(result)
        
        logger.info(f"Analyzed {len(self.policy_results)} policies")
    
    def aggregate_by_jurisdiction(self):
        """Aggregate results by jurisdiction."""
        # Group by jurisdiction
        by_jurisdiction = defaultdict(list)
        for result in self.policy_results:
            by_jurisdiction[result.jurisdiction].append(result)
        
        # Calculate averages
        for jurisdiction, policies in by_jurisdiction.items():
            income_group = policies[0].income_group
            
            jur_result = JurisdictionCapacity(
                jurisdiction=jurisdiction,
                income_group=income_group,
                policy_count=len(policies)
            )
            
            jur_result.avg_institutional = sum(p.institutional_score for p in policies) / len(policies)
            jur_result.avg_enforcement = sum(p.enforcement_score for p in policies) / len(policies)
            jur_result.avg_resources = sum(p.resources_score for p in policies) / len(policies)
            jur_result.avg_operational = sum(p.operational_score for p in policies) / len(policies)
            jur_result.avg_expertise = sum(p.expertise_score for p in policies) / len(policies)
            jur_result.avg_total_capacity = sum(p.total_capacity_score for p in policies) / len(policies)
            
            jur_result.avg_binding = sum(p.binding_score for p in policies) / len(policies)
            jur_result.avg_sector_breadth = sum(p.sector_breadth for p in policies) / len(policies)
            
            # Capacity-ambition gap: negative means more ambition than capacity
            ambition = (jur_result.avg_binding + jur_result.avg_sector_breadth / 10) / 2
            jur_result.capacity_ambition_gap = jur_result.avg_total_capacity - ambition
            
            self.jurisdiction_results[jurisdiction] = jur_result
        
        logger.info(f"Aggregated {len(self.jurisdiction_results)} jurisdictions")
    
    def aggregate_by_income(self) -> Dict:
        """Aggregate by income group."""
        by_income = defaultdict(list)
        for result in self.policy_results:
            by_income[result.income_group].append(result)
        
        income_summary = {}
        for income_group, policies in by_income.items():
            income_summary[income_group] = {
                'policy_count': len(policies),
                'jurisdiction_count': len(set(p.jurisdiction for p in policies)),
                'avg_capacity': sum(p.total_capacity_score for p in policies) / len(policies),
                'avg_institutional': sum(p.institutional_score for p in policies) / len(policies),
                'avg_enforcement': sum(p.enforcement_score for p in policies) / len(policies),
                'avg_resources': sum(p.resources_score for p in policies) / len(policies),
                'avg_operational': sum(p.operational_score for p in policies) / len(policies),
                'avg_expertise': sum(p.expertise_score for p in policies) / len(policies),
                'avg_binding': sum(p.binding_score for p in policies) / len(policies),
            }
        
        return income_summary
    
    def save_results(self, output_dir: str):
        """Save all results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save policy-level results
        policy_file = output_path / 'capacity_by_policy.json'
        with open(policy_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(p) for p in self.policy_results], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved policy results to {policy_file}")
        
        # Save jurisdiction-level results
        jur_file = output_path / 'capacity_by_jurisdiction.json'
        with open(jur_file, 'w', encoding='utf-8') as f:
            json.dump({k: asdict(v) for k, v in self.jurisdiction_results.items()}, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved jurisdiction results to {jur_file}")
        
        # Save income group summary
        income_summary = self.aggregate_by_income()
        income_file = output_path / 'capacity_by_income_group.json'
        with open(income_file, 'w', encoding='utf-8') as f:
            json.dump(income_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved income group results to {income_file}")
        
        # Save CSV for easy viewing
        csv_file = output_path / 'capacity_by_jurisdiction.csv'
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Jurisdiction', 'Income Group', 'Policy Count',
                'Institutional', 'Enforcement', 'Resources', 'Operational', 'Expertise',
                'Total Capacity', 'Binding Score', 'Sector Breadth', 'Capacity-Ambition Gap'
            ])
            for jur, result in sorted(self.jurisdiction_results.items(), 
                                      key=lambda x: x[1].avg_total_capacity, reverse=True):
                writer.writerow([
                    jur, result.income_group, result.policy_count,
                    f"{result.avg_institutional:.3f}",
                    f"{result.avg_enforcement:.3f}",
                    f"{result.avg_resources:.3f}",
                    f"{result.avg_operational:.3f}",
                    f"{result.avg_expertise:.3f}",
                    f"{result.avg_total_capacity:.3f}",
                    f"{result.avg_binding:.3f}",
                    f"{result.avg_sector_breadth:.1f}",
                    f"{result.capacity_ambition_gap:.3f}"
                ])
        logger.info(f"Saved CSV to {csv_file}")
        
        return income_summary
    
    def print_summary(self):
        """Print summary statistics."""
        income_summary = self.aggregate_by_income()
        
        print("\n" + "=" * 70)
        print("IMPLEMENTATION CAPACITY ANALYSIS")
        print("=" * 70)
        
        print("\n📊 BY INCOME GROUP:")
        print("-" * 70)
        print(f"{'Income Group':<20} {'Countries':<12} {'Policies':<10} {'Capacity':<10} {'Enforcement':<12}")
        print("-" * 70)
        
        for group in ['High Income', 'Upper Middle', 'Lower Middle', 'Low Income', 'Unclassified']:
            if group in income_summary:
                s = income_summary[group]
                print(f"{group:<20} {s['jurisdiction_count']:<12} {s['policy_count']:<10} "
                      f"{s['avg_capacity']:.3f}     {s['avg_enforcement']:.3f}")
        
        print("\n📈 TOP 10 JURISDICTIONS BY CAPACITY:")
        print("-" * 70)
        top_10 = sorted(self.jurisdiction_results.items(), 
                       key=lambda x: x[1].avg_total_capacity, reverse=True)[:10]
        for rank, (jur, result) in enumerate(top_10, 1):
            print(f"{rank:2}. {jur:<30} {result.avg_total_capacity:.3f} ({result.income_group})")
        
        print("\n⚠️  LARGEST CAPACITY-AMBITION GAPS (under-resourced):")
        print("-" * 70)
        gaps = sorted(self.jurisdiction_results.items(), 
                     key=lambda x: x[1].capacity_ambition_gap)[:10]
        for jur, result in gaps:
            if result.capacity_ambition_gap < 0:
                print(f"   {jur:<30} Gap: {result.capacity_ambition_gap:.3f} ({result.income_group})")
        
        print("\n" + "=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze AI governance implementation capacity')
    parser.add_argument('--enriched', default='data/oecd/enriched/oecd_enriched_20260127_203406.json')
    parser.add_argument('--corpus', default='data/corpus/corpus_master_20260128.json')
    parser.add_argument('--output', default='data/analysis/capacity')
    args = parser.parse_args()
    
    analyzer = CapacityAnalyzer(args.enriched, args.corpus)
    analyzer.analyze_all()
    analyzer.aggregate_by_jurisdiction()
    analyzer.save_results(args.output)
    analyzer.print_summary()


if __name__ == '__main__':
    main()
